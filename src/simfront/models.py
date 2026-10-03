"""Real model configurations, instantiated without weights.

The JSON files in ``configs/`` hold the architectural fields of published Hugging Face
configs (each records its source URL). ``build`` instantiates the model on the meta
device: every parameter has a shape and a dtype and no storage, so Llama-3-70B costs
no memory. ``analytic_spec`` gives the same model as Disaggregated_Inference_Sim's
``ModelSpec``, so an operator trace can be checked against that simulator's closed-form
FLOP counts.
"""

from __future__ import annotations

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
