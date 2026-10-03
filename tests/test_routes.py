"""The four front ends agree on the arithmetic, and the ONNX export is a faithful model."""

import numpy as np
import pytest
import torch

from simfront import models
from simfront.capture import export_onnx, trace_compile, trace_dispatch, trace_export, trace_onnx

T = 32


def all_routes(cfg, tmp_path, device="cpu"):
    m = models.build(cfg, device=device, dtype=torch.float32)
    x = models.tokens(1, T, device)
    out = {"dispatch": trace_dispatch(m, x, use_cache=False)[0] if device == "cpu" else trace_dispatch(m, x)[0],
           "export": trace_export(m, (x,), {"use_cache": False}),
           "compile": trace_compile(m, (x,), {"use_cache": False})[0]}
    path = tmp_path / "m.onnx"
    out["onnx"] = trace_onnx(path, *export_onnx(m, (x,), path, {"use_cache": False}, weights=False))
    return m, x, out


def rope(cfg):
    """The rotary-frequency bmm: position-only arithmetic that ONNX constant-folds (static shapes)."""
    return 2 * (cfg.hidden_size // cfg.num_attention_heads // 2) * T if cfg.model_type == "llama" else 0


def qk_av_and_weights(t, cfg=None):
    """Matmul FLOPs, plus fused attention ops' QK^T and AV (their softmax share removed); ONNX
    traces get the folded rotary bmm back so that every route is compared like for like."""
    softmax = sum(5 * (o.outputs[0].numel // o.inputs[0].shape[-1]) * o.inputs[1].shape[-2]
                  for o in t.ops if o.category == "attention")
    folded = rope(cfg) if (cfg is not None and t.route == "onnx") else 0
    return t.flops_of("matmul") + t.flops_of("attention") - softmax + folded


@pytest.mark.parametrize("cfg", [models.tiny_llama(), models.tiny_gpt2()], ids=["llama", "gpt2"])
def test_routes_agree_on_flops_and_weights(cfg, tmp_path):
    _, _, tr = all_routes(cfg, tmp_path)
    # On a real CPU, dispatch and compile keep the fused attention kernel; export and ONNX decompose it.
    got = {k: qk_av_and_weights(t, cfg) for k, t in tr.items()}
    assert len(set(got.values())) == 1, got
    # With static shapes the positions are constants, so ONNX folds GPT-2's position-embedding lookup
    # (a Gather of constant rows), as a runtime would at load time: those rows are no longer read.
    folded_pos = T * cfg.n_embd * 4 if cfg.model_type == "gpt2" else 0
    assert tr["onnx"].weight_bytes + folded_pos == tr["dispatch"].weight_bytes
    assert len({t.weight_bytes for k, t in tr.items() if k != "onnx"}) == 1
    for t in tr.values():
        assert t.unknown() == {}, (t.route, t.unknown())


def test_routes_agree_on_the_meta_device(tmp_path):
    cfg = models.tiny_llama()
    m = models.build(cfg)
    x = models.tokens(1, T)
    d = trace_dispatch(m, x)[0]
    e = trace_export(m, (x,), {"use_cache": False})
    c = trace_compile(m, (x,), {"use_cache": False})[0]
    path = tmp_path / "m.onnx"
    o = trace_onnx(path, *export_onnx(m, (x,), path, {"use_cache": False}, weights=False))
    assert d.flops_of("matmul") == e.flops_of("matmul") == c.flops_of("matmul") == o.flops_of("matmul") + rope(cfg)
    assert d.weight_bytes == e.weight_bytes == c.weight_bytes == o.weight_bytes


def test_onnx_export_runs_in_onnxruntime_and_matches_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    m = models.build(models.tiny_llama(), device="cpu", dtype=torch.float32)
    x = torch.randint(0, 1000, (1, T))
    path = tmp_path / "w.onnx"
    names, params = export_onnx(m, (x,), path, {"use_cache": False}, weights=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {names[0]: x.numpy()})[0]
    with torch.no_grad():
        ref = m(x, use_cache=False).logits.numpy()
    assert np.abs(got - ref).max() < 1e-4
    # With or without weights in the file, the walker sees the same per-inference arithmetic.
    path2 = tmp_path / "nw.onnx"
    a = trace_onnx(path, names, params)
    b = trace_onnx(path2, *export_onnx(m, (x,), path2, {"use_cache": False}, weights=False))
    assert b.flops_of("matmul") == a.flops_of("matmul")
    assert a.weight_bytes == b.weight_bytes
    lm_head = [o for o in b.ops if o.name == "onnx.MatMul"][-1]       # Transpose(W) was folded: still a weight
    assert lm_head.weight_bytes == 1000 * 256 * 4


class Block(torch.nn.Module):
    """Attention + MLP in plain PyTorch, with no data-dependent Python branches."""

    def __init__(self, d=64, h=4):
        super().__init__()
        self.h, self.qkv, self.o = h, torch.nn.Linear(d, 3 * d), torch.nn.Linear(d, d)
        self.up, self.down, self.norm = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d), torch.nn.LayerNorm(d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(torch.nn.functional.gelu(self.up(x)))


def signature(t):
    return [(o.name, [i.shape for i in o.inputs], [i.shape for i in o.outputs]) for o in t.ops
            if not o.name.startswith("prim.")]


def test_fake_tensors_trace_exactly_what_real_tensors_run():
    torch.manual_seed(0)
    real = trace_dispatch(Block(), torch.randn(2, 16, 64))[0]
    with models.fake_cpu():
        fake = trace_dispatch(Block(), torch.randn(2, 16, 64))[0]
    assert signature(real) == signature(fake)
    assert any(o.category == "attention" for o in fake.ops)


def test_transformers_takes_a_different_path_when_it_sees_tracing():
    """transformers checks for tracing (fake, meta, export, compile) and then builds an explicit mask
    and repeats the KV heads; in eager mode on real data it inspects the mask with .item() and uses
    SDPA's is_causal and native GQA. Same arithmetic, different memory traffic: a trace shows the
    traced code path."""
    cfg = models.tiny_llama()
    real = trace_dispatch(models.build(cfg, device="cpu"), models.tokens(1, T, "cpu"), use_cache=False)[0]
    with models.fake_cpu():
        fake = trace_dispatch(models.build(cfg, device="cpu"), models.tokens(1, T, "cpu"), use_cache=False)[0]
    assert qk_av_and_weights(real) == qk_av_and_weights(fake)
    assert "aten._local_scalar_dense" in real.counts() and "aten._local_scalar_dense" not in fake.counts()
    kv_heads = lambda t: {o.inputs[1].shape[1] for o in t.ops if o.category == "attention"}  # noqa: E731
    assert kv_heads(real) == {2} and kv_heads(fake) == {4}          # native GQA vs repeat_kv
    assert fake.bytes > real.bytes


def test_compile_backend_returns_correct_outputs():
    from simfront.capture import SimBackend

    m = models.build(models.tiny_llama(), device="cpu", dtype=torch.float32)
    x = torch.randint(0, 1000, (2, T))
    with torch.no_grad():
        ref = m(x, use_cache=False).logits
        torch._dynamo.reset()
        be = SimBackend()
        got = torch.compile(m, backend=be)(x, use_cache=False).logits
    assert torch.equal(ref, got)
    assert len(be.graphs) >= 1 and any(o.name == "aten.mm" for o in be.graphs[0])
