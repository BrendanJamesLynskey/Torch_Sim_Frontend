"""How an ONNX Runtime execution provider takes part of a graph, in miniature.

ONNX Runtime (ORT) runs a model through an ordered list of execution providers (EPs).
When a session is created, ORT asks each EP in priority order which nodes it can run
(``IExecutionProvider::GetCapability``); the EP answers with groups of nodes. ORT
partitions the graph accordingly, lets each EP fuse and compile its groups (``Compile``
turns each group into one kernel), and gives every unclaimed node to the CPU EP, which
supports every standard operator. A hardware backend is an EP: its ``GetCapability`` says
what the device can run.

A real EP is C++ built against ORT's headers (recent ORT releases can also load an EP
from a separate plugin library at run time; see ORT's execution-provider documentation).
Building one is out of scope here. This module reproduces the
*partitioning* in Python, so a simulator can answer the question an EP's author asks first:
for this model, what runs on the device, in how many pieces, and how much data crosses
between device and host?

* :func:`claim` is a ``GetCapability``: it claims the nodes whose operator types the device
  supports and groups claimed nodes that are connected into one subgraph each.
* :func:`ort_providers` runs the model in a real ORT session with profiling on and reads,
  from the profile, which EP ran each node: the ground truth of what ORT did.
* :func:`simulate` costs a partitioned model: claimed subgraphs on the accelerator model,
  everything else on a host roofline, and a link transfer for every tensor that crosses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import onnx

from ..cost import HOST_CPU, Roofline
from ..trace import Trace


@dataclass
class Partition:
    nodes: list[tuple[str, str, str]]       # (node name, op type, "device" or "host")
    subgraphs: list[list[str]]              # claimed node names, one list per fused subgraph

    @property
    def claimed(self) -> int:
        return sum(1 for _, _, side in self.nodes if side == "device")


def claim(model: onnx.ModelProto | str | Path, supported: set[str]) -> Partition:
    """Claim nodes by operator type and group connected claimed nodes (union-find over edges)."""
    m = onnx.load(str(model)) if not isinstance(model, onnx.ModelProto) else model
    nodes = list(m.graph.node)
    names = [n.name or f"{n.op_type}_{i}" for i, n in enumerate(nodes)]
    produced = {o: i for i, n in enumerate(nodes) for o in n.output}
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    on = [n.op_type in supported for n in nodes]
    for i, n in enumerate(nodes):
        if not on[i]:
            continue
        for x in n.input:
            j = produced.get(x)
            if j is not None and on[j]:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i in range(len(nodes)):
        if on[i]:
            groups.setdefault(find(i), []).append(names[i])
    return Partition([(names[i], n.op_type, "device" if on[i] else "host") for i, n in enumerate(nodes)],
                     list(groups.values()))


def ort_providers(model_path: str | Path, feeds: dict, providers: list[str] | None = None) -> list[dict]:
    """Run one inference with ORT profiling on; return each node's name, op type and provider."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL     # keep the graph as exported
    sess = ort.InferenceSession(str(model_path), so, providers=providers or ["CPUExecutionProvider"])
    sess.run(None, feeds)
    prof = Path(sess.end_profiling())
    events = json.loads(prof.read_text())
    prof.unlink(missing_ok=True)
    out, seen = [], set()
    for e in events:
        a = e.get("args", {})
        if e.get("cat") == "Node" and "provider" in a and e["name"].endswith("_kernel_time"):
            node = e["name"][: -len("_kernel_time")]
            if node not in seen:
                seen.add(node)
                out.append({"node": node, "op_type": a.get("op_name"), "provider": a["provider"]})
    return out


@dataclass
class SplitCost:
    device_s: float
    host_s: float
    link_s: float
    crossings: int
    crossing_bytes: int

    @property
    def total(self) -> float:
        return self.device_s + self.host_s + self.link_s


def simulate(trace: Trace, supported_categories: set[str], cfg, host: Roofline = HOST_CPU,
             link_bw: float = 64e9, link_latency: float = 1e-6) -> tuple[SplitCost, "object"]:
    """Claimed categories on the accelerator model (``cfg``), the rest on ``host``, executed in
    trace order; every activation that crosses between the two pays one link transfer."""
    from .fastpath import run as run_fast
    from .lower import lower
    from .metrics import report
    from .sim import simulate as run_simpy

    dev_ops, host_s, where = [], 0.0, {}
    link_s, crossings, xbytes = 0.0, 0, 0
    for o in trace.ops:
        if o.category in ("view", "unknown"):
            for t in o.outputs:
                where[t.tid] = where.get(o.inputs[0].tid, "device") if o.inputs else "device"
            continue
        side = "device" if o.category in supported_categories else "host"
        for t in o.inputs:
            if not t.param and where.get(t.tid, "device") != side:
                link_s += link_latency + t.nbytes / link_bw
                crossings += 1
                xbytes += t.nbytes
                where[t.tid] = side
        if side == "device":
            dev_ops.append(o)
        else:
            host_s += host.op_cost(o).time
        for t in o.outputs:
            where[t.tid] = side
    rep = None
    dev_s = 0.0
    if dev_ops:
        prog = lower(Trace(trace.model, trace.route, dev_ops, trace.workload), cfg)
        tm = run_fast(prog) if cfg.dram_channels >= 2 else run_simpy(prog)[0]
        rep = report(prog, tm)
        dev_s = rep.makespan
    return SplitCost(dev_s, host_s, link_s, crossings, xbytes), rep
