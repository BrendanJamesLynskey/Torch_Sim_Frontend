"""Front end 3: a torch.compile backend that costs each graph and then runs it eagerly.

TorchDynamo captures graphs from ordinary Python as the model runs and passes each to a
backend: any callable taking an FX ``GraphModule`` and example inputs and returning a
callable. Dynamo's graph is at the torch level (``linear``, ``scaled_dot_product_attention``),
so this backend hands it to AOTAutograd, which lowers it to ATen, and walks that graph
with the same cost rules as the other front ends. It returns the graph's own ``forward``,
so the user's program still produces correct outputs: the simulator rides along.

Graph breaks (Python that Dynamo cannot capture) split a model into several graphs; each
reaches the backend, and ``SimBackend.graphs`` keeps them in order.
"""

from __future__ import annotations

import time

import torch
from torch._dynamo.backends.common import aot_autograd

from ..trace import Op, Trace
from .fx_walk import walk_graph


class SimBackend:
    """``torch.compile(model, backend=SimBackend())``: records an operator trace per captured graph."""

    def __init__(self):
        self.graphs: list[list[Op]] = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        # Dynamo names lifted parameters after their module path ("..._parameters_weight_");
        # AOTAutograd keeps the input order, so the i-th placeholder below is the same tensor.
        is_param = [n.op == "placeholder" and "_parameters_" in n.name for n in gm.graph.nodes
                    if n.op == "placeholder"]

        def fw_compiler(fgm: torch.fx.GraphModule, _inputs):
            ph = [n for n in fgm.graph.nodes if n.op == "placeholder"]
            params = {n.name for n, p in zip(ph, is_param, strict=True) if p}
            self.graphs.append(walk_graph(fgm.graph, params))
            return fgm.forward

        return aot_autograd(fw_compiler=fw_compiler)(gm, example_inputs)


def trace_compile(model: torch.nn.Module, args: tuple, kwargs: dict | None = None, *, model_name: str = "",
                  workload: dict | None = None) -> tuple[Trace, int]:
    """Compile with :class:`SimBackend`, run once, return (trace of all graphs, number of graphs)."""
    torch._dynamo.reset()
    t0 = time.perf_counter()
    backend = SimBackend()
    compiled = torch.compile(model, backend=backend, dynamic=False)
    with torch.no_grad():
        compiled(*args, **(kwargs or {}))
    ops = [o for g in backend.graphs for o in g]
    return Trace(model_name or type(model).__name__, "compile", ops, workload or {},
                 time.perf_counter() - t0), len(backend.graphs)
