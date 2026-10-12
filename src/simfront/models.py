"""Real model configurations, instantiated without weights.

The JSON files in ``configs/`` hold the architectural fields of published Hugging Face
configs (each records its source URL). ``build`` instantiates the model on the meta
device: every parameter has a shape and a dtype and no storage, so Llama-3-70B costs
no memory. ``analytic_spec`` gives the same model as Disaggregated_Inference_Sim's
``ModelSpec``, so an operator trace can be checked against that simulator's closed-form
FLOP counts.
"""

from __future__ import annotations

import functools
import json
from contextlib import contextmanager
from importlib import resources

import torch

MODELS = ["llama3-8b", "llama3-70b", "mistral-7b", "qwen2.5-0.5b", "gpt2"]


def config_fields(name: str) -> dict:
    return json.loads(resources.files("simfront.configs").joinpath(f"{name}.json").read_text())


def config(name: str):
    from transformers import AutoConfig

    fields = {k: v for k, v in config_fields(name).items() if not k.startswith("_")}
    cfg = AutoConfig.for_model(fields.pop("model_type"), **fields)
    cfg._attn_implementation = "sdpa"
    return cfg


def tiny_llama(**overrides):
    """A two-layer Llama (GQA, SwiGLU, RoPE) small enough to run for real on a CPU in tests."""
    from transformers import LlamaConfig

    kw = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, vocab_size=1000)
    kw.update(overrides)
    cfg = LlamaConfig(**kw)
    cfg._attn_implementation = "sdpa"
    return cfg


def tiny_gpt2(**overrides):
    """A two-layer GPT-2 (LayerNorm, GELU, Conv1D projections with biases, learned positions)."""
    from transformers import GPT2Config

    kw = dict(n_embd=128, n_layer=2, n_head=4, vocab_size=1000, n_positions=256)
    kw.update(overrides)
    cfg = GPT2Config(**kw)
    cfg._attn_implementation = "sdpa"
    return cfg


class TinyCNN(torch.nn.Module):
    """A small image classifier: three conv-batchnorm-ReLU blocks, max pooling, a linear head.

    The convolution workload for the accelerator model (``simfront.accel``): small enough to
    export to ONNX with weights and run in ONNX Runtime in a test.
    """

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64), classes: int = 10):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers += [torch.nn.Conv2d(c_in, c, 3, padding=1), torch.nn.BatchNorm2d(c), torch.nn.ReLU(),
                       torch.nn.MaxPool2d(2)]
            c_in = c
        self.features = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(c_in, classes)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def tiny_cnn(seed: int = 0, **kw) -> TinyCNN:
    torch.manual_seed(seed)
    return TinyCNN(**kw).eval()


def image(batch: int = 1, size: int = 32, device: str = "cpu") -> torch.Tensor:
    return torch.zeros(batch, 3, size, size, device=device)


def build(cfg, device: str = "meta", dtype: torch.dtype = torch.bfloat16, seed: int = 0) -> torch.nn.Module:
    """``AutoModelForCausalLM.from_config`` on ``device``; ``cfg`` is a config or a registry name."""
    from transformers import AutoModelForCausalLM

    if isinstance(cfg, str):
        cfg = config(cfg)
    torch.manual_seed(seed)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    return model.eval()


@contextmanager
def fake_cpu():
    """Build and trace inside this to see the operators PyTorch would run on a CPU, with no data.

    Under a ``FakeTensorMode`` tensors claim to be on the CPU, so ``scaled_dot_product_attention``
    dispatches to the fused CPU flash-attention kernel (one operator) instead of the math
    decomposition the meta device gets (``bmm``, ``_safe_softmax``, ``bmm``, with the full score
    matrix in memory). transformers also recognises fake tensors as tracing, so the cache can be
    off. Build the model and its inputs with ``device="cpu"`` inside the block.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        yield


def tokens(batch: int, length: int, device: str = "meta") -> torch.Tensor:
    return torch.zeros(batch, length, dtype=torch.long, device=device)


def decode_inputs(model: torch.nn.Module, context: int, batch: int = 1, device: str = "meta") -> dict:
    """Arguments for one decode step after a ``context``-token prompt.

    The prompt is run first, outside any trace, to fill the KV cache (on the meta device
    this costs nothing). Note: with the cache disabled, transformers' mask code calls
    ``.item()`` on a data-dependent tensor, which the meta device cannot do, so prefill
    traces keep the cache on too.
    """
    with torch.no_grad():
        out = model(tokens(batch, context, device))
    return {"input_ids": tokens(batch, 1, device), "past_key_values": out.past_key_values}


def analytic_spec(name: str):
    """Disaggregated_Inference_Sim's ``ModelSpec`` for a SwiGLU decoder in the registry."""
    from disagg_sim.hardware import LLAMA3_8B, LLAMA3_70B, ModelSpec

    if name == "llama3-8b":
        return LLAMA3_8B
    if name == "llama3-70b":
        return LLAMA3_70B
    c = config_fields(name)
    if c["model_type"] not in ("llama", "mistral", "qwen2"):
        raise ValueError(f"{name}: not a SwiGLU decoder")
    return ModelSpec(name, n_layers=c["num_hidden_layers"], d_model=c["hidden_size"], n_heads=c["num_attention_heads"],
                     n_kv_heads=c["num_key_value_heads"], d_ff=c["intermediate_size"], vocab=c["vocab_size"])


@functools.cache
def rotary_runs_as_matmul(model_type: str = "llama") -> bool:
    """Whether the installed transformers computes ``model_type``'s rotary frequencies with a matmul.

    The rotary embedding needs the outer product of the positions and the inverse frequencies. Up to
    transformers 5.18 the Llama, Mistral and Qwen2 code wrote it as a batched matmul with an inner
    dimension of 1 (``inv_freq[:, None] @ position_ids[None, :]``), which a trace counts as a matmul of
    2 x positions x head_dim/2 FLOPs. From 5.19 it is a broadcast multiply
    (``position_ids[..., None] * inv_freq``): the same products, one elementwise multiply each, and no
    matmul. Rather than test a version number, this traces the rotary module of a one-layer model of
    that type once (on the meta device, which costs nothing) and looks at what actually ran.
    """
    from transformers import AutoConfig, AutoModel

    from simfront.capture import trace_dispatch

    cfg = AutoConfig.for_model(model_type, hidden_size=16, intermediate_size=16, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=1, vocab_size=8)
    with torch.device("meta"):
        rotary = AutoModel.from_config(cfg).rotary_emb
    x = torch.zeros(1, 4, 16, device="meta")
    positions = torch.arange(4, device="meta")[None]
    return trace_dispatch(rotary, x, positions)[0].flops_of("matmul") > 0


def rotary_matmul_flops(head_dim: int, positions: int, model_type: str = "llama") -> int:
    """The matmul FLOPs a trace shows for computing the rotary frequencies of ``positions`` positions.

    2 x positions x head_dim/2 when the installed transformers uses a matmul for it, otherwise 0 (it is
    then ``positions x head_dim/2`` elementwise multiplies). See :func:`rotary_runs_as_matmul`.
    """
    return 2 * (head_dim // 2) * positions if rotary_runs_as_matmul(model_type) else 0
