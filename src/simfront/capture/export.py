"""Front end 2: torch.export, then walk the ExportedProgram's graph.

``torch.export`` traces a model into one ATen-level FX graph with shape metadata on every
node, without running any arithmetic (it uses fake tensors), so it also works on the meta
device. ``run_decompositions()`` lowers it to Core ATen: ``linear`` becomes ``permute`` +
``mm``/``addmm``, and ``scaled_dot_product_attention`` becomes ``bmm`` + ``_softmax``. That
shrinks the set of operators a cost model must support, which is the point.
"""

from __future__ import annotations

import time

import torch

from ..trace import Trace
from .fx_walk import walk_graph


def trace_export(model: torch.nn.Module, args: tuple, kwargs: dict | None = None, *, decompose: bool = True,
                 model_name: str = "", workload: dict | None = None) -> Trace:
    t0 = time.perf_counter()
    ep = torch.export.export(model, args, kwargs=kwargs or {}, strict=False)
    if decompose:
        ep = ep.run_decompositions()
    params = set(ep.graph_signature.inputs_to_parameters)
    ops = walk_graph(ep.graph, params)
    return Trace(model_name or type(model).__name__, "export" if decompose else "export-predispatch", ops,
                 workload or {}, time.perf_counter() - t0)
