"""Walk an ATen-level FX graph (from torch.export or AOTAutograd) into trace operators.

Each ``call_function`` node carries a fake tensor in ``node.meta["val"]``: shape and dtype,
no data. Tuple-returning operators (``native_layer_norm``) are followed by ``getitem``
nodes, which are resolved to the right output rather than counted.
"""

from __future__ import annotations

import operator

import torch
from torch.utils._pytree import tree_flatten

from ..rules import apply_aten_rule
from ..trace import Op, TensorMeta
from .dispatch import VIEW_PACKETS


def _metas(val, tid: str, param: bool) -> list[TensorMeta]:
    if isinstance(val, torch.Tensor):
        return [TensorMeta(tid, tuple(int(s) for s in val.shape), val.element_size(), param)]
    if isinstance(val, (tuple, list)):
        return [TensorMeta(f"{tid}.{i}", tuple(int(s) for s in v.shape), v.element_size(), param)
                for i, v in enumerate(val) if isinstance(v, torch.Tensor)]
    return []


def walk_graph(graph: torch.fx.Graph, params: set[str]) -> list[Op]:
    """Operators of an ATen FX graph; ``params`` names the placeholders that are model weights."""
    tensors: dict[str, TensorMeta | list[TensorMeta]] = {}
    ops: list[Op] = []
    for n in graph.nodes:
        val = n.meta.get("val")
        if n.op == "placeholder":
            m = _metas(val, n.name, n.name in params)
            tensors[n.name] = m[0] if len(m) == 1 else m
            continue
        if n.op != "call_function":
            continue
        if n.target is operator.getitem:                          # select one output of a tuple op
            parent, i = n.args
            src = tensors.get(parent.name)
            if isinstance(src, list) and i < len(src):
                tensors[n.name] = src[i]
            continue
        flat, _ = tree_flatten((n.args, n.kwargs))
        ins = []
        for a in flat:
            if isinstance(a, torch.fx.Node):
                t = tensors.get(a.name)
                if isinstance(t, TensorMeta):
                    ins.append(t)
                elif isinstance(t, list):
                    ins.extend(t)
        name = "aten." + str(n.target).split(".")[1] if str(n.target).startswith("aten.") else str(n.target)
        param_view = name.split(".", 1)[-1] in VIEW_PACKETS and any(t.param for t in ins)
        outs = _metas(val, n.name, param_view)
        tensors[n.name] = outs[0] if len(outs) == 1 else outs
        attrs = {k: v for k, v in n.kwargs.items() if isinstance(v, (int, float, bool, str))}
        ops.append(apply_aten_rule(Op(name, ins, outs, attrs)))
    return ops
