"""Front end 1: record every ATen operator a model runs, with a TorchDispatchMode.

Every PyTorch operator passes through the dispatcher, and a ``TorchDispatchMode`` sees
each call after autograd and Python-level decompositions, with real argument tensors.
On the meta device those tensors have shapes and dtypes but no storage, so a model of
any size can be traced on a laptop: Llama-3-70B's 70 billion parameters take no memory.

What the trace shows is what PyTorch would execute *on this device*. On the meta device
``scaled_dot_product_attention`` has no fused kernel, so it appears decomposed into
``bmm``, ``_safe_softmax`` and a mask; a CUDA trace would show one flash-attention call.
"""

from __future__ import annotations

import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from ..rules import ATEN_CATEGORY, apply_aten_rule
from ..trace import Op, TensorMeta, Trace

VIEW_PACKETS = {k for k, v in ATEN_CATEGORY.items() if v == "view"}


class OpTrace(TorchDispatchMode):
    """Records each ATen call as an :class:`Op`, tracking tensor identity and parameters."""

    def __init__(self, params=()):
        super().__init__()
        self.ops: list[Op] = []
        self._ids: dict[int, str] = {}
        self._keep: list[torch.Tensor] = []          # keep tensors alive so id() is never reused
        self._param = {id(p) for p in params}

    def _meta(self, t: torch.Tensor, param: bool = False) -> TensorMeta:
        k = id(t)
        if k not in self._ids:
            self._ids[k] = f"t{len(self._ids)}"
            self._keep.append(t)
            if param:
                self._param.add(k)
        return TensorMeta(self._ids[k], tuple(int(s) for s in t.shape), t.element_size(), k in self._param)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = str(func.overloadpacket)                # e.g. "aten.mm"
        flat_in, _ = tree_flatten((args, kwargs))
        flat_out, _ = tree_flatten(out)
        ins = [self._meta(t) for t in flat_in if isinstance(t, torch.Tensor)]
        # A view of a parameter (W.t() before mm) is still weight traffic.
        is_view = name.split(".", 1)[1] in VIEW_PACKETS
        param_view = is_view and any(t.param for t in ins)
        outs = [self._meta(t, param=param_view) for t in flat_out if isinstance(t, torch.Tensor)]
        attrs = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, bool, str))}
        self.ops.append(apply_aten_rule(Op(name, ins, outs, attrs)))
        return out


def trace_dispatch(model: torch.nn.Module, *args, model_name: str = "", workload: dict | None = None,
                   **kwargs) -> tuple[Trace, object]:
    """Run ``model(*args, **kwargs)`` under :class:`OpTrace`; returns (trace, model output)."""
    t0 = time.perf_counter()
    with torch.no_grad(), OpTrace(model.parameters()) as t:
        out = model(*args, **kwargs)
    return Trace(model_name or type(model).__name__, "dispatch", t.ops, workload or {},
                 time.perf_counter() - t0), out
