"""Real model configurations traced on the meta device, checked against closed forms and PyTorch's own counter."""

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from simfront import models
from simfront.capture import trace_dispatch


def rope_flops(spec, tokens):
    """The rotary-embedding frequency outer product, done once per forward as a bmm."""
    return 2 * (spec.head_dim // 2) * tokens


@pytest.mark.req("SF-01")
def test_llama3_8b_trace_shapes(llama8b_prefill):
    mms = [o for o in llama8b_prefill.ops if o.name == "aten.mm"]
    assert len(mms) == 7 * 32 + 1                       # q, k, v, o, gate, up, down per layer + LM head
    shapes = [(o.inputs[0].shape, o.inputs[1].shape) for o in mms[:7]]
    assert shapes == [((512, 4096), (4096, 4096)), ((512, 4096), (4096, 1024)), ((512, 4096), (4096, 1024)),
                      ((512, 4096), (4096, 4096)), ((512, 4096), (4096, 14336)), ((512, 4096), (4096, 14336)),
                      ((512, 14336), (14336, 4096))]
    assert mms[-1].outputs[0].shape == (512, 128256)
    assert all(o.inputs[1].param for o in mms)          # every mm reads a weight
    assert llama8b_prefill.unknown() == {}


@pytest.mark.req("SF-02", "SF-03")
@pytest.mark.parametrize("name", ["llama3-8b", "llama3-70b", "mistral-7b"])
def test_prefill_matmul_flops_equal_closed_form(name):
    spec, T = models.analytic_spec(name), 256
    m = models.build(name)
    tr = trace_dispatch(m, models.tokens(1, T))[0]
    weights = 2 * spec.matmul_params * T                # Disaggregated_Inference_Sim's prefill term
    attention = 4 * spec.n_layers * spec.d_model * T * T   # QK^T and AV, unmasked
    assert tr.flops_of("matmul") == weights + attention + rope_flops(spec, T)
    norms = (2 * spec.n_layers + 1) * spec.d_model
    assert tr.weight_bytes == 2 * (spec.matmul_params + T * spec.d_model + norms)
    assert all(p.is_meta for p in m.parameters())       # nothing was allocated


@pytest.mark.req("SF-03")
def test_qwen_biases_are_counted():
    spec, T = models.analytic_spec("qwen2.5-0.5b"), 128
    tr = trace_dispatch(models.build("qwen2.5-0.5b"), models.tokens(1, T))[0]
    kv = spec.n_kv_heads * spec.head_dim
    bias = T * spec.n_layers * (spec.d_model + 2 * kv)  # q, k and v projections carry biases
    assert tr.flops_of("matmul") == 2 * spec.matmul_params * T + 4 * spec.n_layers * spec.d_model * T * T \
        + rope_flops(spec, T) + bias


@pytest.mark.req("SF-05")
def test_decode_step_matches_closed_form():
    spec, C = models.analytic_spec("llama3-8b"), 1024
    m = models.build("llama3-8b")
    tr = trace_dispatch(m, **models.decode_inputs(m, C))[0]
    from disagg_sim.hardware import H100_SXM, CostModel
    analytic = CostModel(spec, H100_SXM).decode([C])
    # The new token attends to C cached keys and to itself: C + 1 keys. This test found that the closed form
    # counted C; Disaggregated_Inference_Sim was corrected on 2026-10-03 and now agrees, up to the rotary matmul.
    assert tr.flops_of("matmul") == analytic.flops + rope_flops(spec, 1)
    # Weights read: the layers, the LM head and one embedding row (the closed form once charged the whole
    # table, 1.05 GB per step: also found here), plus the RMSNorm weights, which the closed form does not model.
    norms = 2 * (2 * spec.n_layers + 1) * spec.d_model
    assert tr.weight_bytes == spec.weight_bytes_read(1) + norms


@pytest.mark.req("SF-03")
def test_matches_pytorch_flop_counter(llama8b_prefill):
    m = models.build("llama3-8b")
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(models.tokens(1, 512))
    assert fc.get_total_flops() == llama8b_prefill.flops_of("matmul")


@pytest.mark.req("SF-04")
def test_fake_cpu_gives_fused_attention_with_the_same_arithmetic(llama8b_prefill):
    with models.fake_cpu():
        m = models.build("llama3-8b", device="cpu")
        tr = trace_dispatch(m, models.tokens(1, 512, "cpu"), use_cache=False)[0]
    att = [o for o in tr.ops if o.category == "attention"]
    assert len(att) == 32 and att[0].name == "aten._scaled_dot_product_flash_attention_for_cpu"
    softmax = 5 * 32 * 32 * 512 * 512                   # the fused op's softmax share, 5 per score
    meta_bmm = llama8b_prefill.flops_of("matmul") - tr.flops_of("matmul")
    assert tr.flops_of("attention") - softmax == meta_bmm
    # PyTorch's FlopCounterMode has no formula for the CPU flash-attention kernel: it drops these FLOPs.
    with models.fake_cpu():
        m = models.build("llama3-8b", device="cpu")
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(models.tokens(1, 512, "cpu"), use_cache=False)
    assert fc.get_total_flops() == tr.flops_of("matmul")


@pytest.mark.req("SF-13")
def test_capture_of_llama3_8b_prefill_is_fast():
    """SF-13: under 5 s (it takes well under 1 s here); the CI gate also tracks it against a baseline."""
    m = models.build("llama3-8b")
    tr = trace_dispatch(m, models.tokens(1, 2048))[0]
    assert tr.capture_s < 5.0
