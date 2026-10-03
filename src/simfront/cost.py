"""Pluggable accelerator cost models for operator traces.

A cost model takes a :class:`~simfront.trace.Trace` and returns a :class:`CostReport`.
Two are provided, both starting from the roofline of LLM Inference Simulators
(Disaggregated_Inference_Sim's ``Accelerator`` specs, with the same efficiency derating):

* :class:`Roofline`: every operator on one device, time = max(FLOPs / FLOP rate,
  bytes / byte rate), summed over operators. ``memory="fused"`` is the ideal-fusion
  bound: only matmul, attention and gather operators touch memory; everything else is
  assumed to stay on chip. The truth for a real compiler lies between the two.
* :class:`Offload`: an accelerator that supports only some operator categories (say, a
  matmul engine), with everything else on a host processor, and data moved over a link
  whenever an operator needs a tensor that lives on the other side. This is where
  operator coverage stops being a count and becomes time.

Device numbers come from Disaggregated_Inference_Sim and are datasheet-level or
illustrative, as labelled there; the host and link presets below are illustrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from .trace import Op, Trace

MEMORY_OPS = {"matmul", "attention", "gather"}     # what still touches memory under ideal fusion


@dataclass(frozen=True)
class OpCost:
    time: float
    bound: str          # "compute", "memory", "free" (views, unknown) or "transfer"
    unit: str           # "accel", "host" or "-"
    transfer: float = 0.0


@dataclass
class CostReport:
    model: str
    trace: Trace
    costs: list[OpCost]
    notes: dict = field(default_factory=dict)

    @property
    def time(self) -> float:
        return sum(c.time + c.transfer for c in self.costs)

    def share(self, pred: Callable[[Op, OpCost], bool]) -> float:
        tot = self.time
        pairs = zip(self.trace.ops, self.costs, strict=True)
        return sum(c.time + c.transfer for o, c in pairs if pred(o, c)) / tot if tot else 0.0

    def by_category(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for o, c in zip(self.trace.ops, self.costs, strict=True):
            d = out.setdefault(o.category, {"ops": 0, "flops": 0, "bytes": 0, "time": 0.0, "compute": 0.0,
                                            "memory": 0.0})
            d["ops"] += 1
            d["flops"] += o.flops
            d["bytes"] += o.bytes
            d["time"] += c.time + c.transfer
            if c.bound in ("compute", "memory"):
                d[c.bound] += c.time
        return out

    def bound_split(self) -> dict[str, float]:
        """Share of total time spent in compute-bound ops, memory-bound ops and transfers."""
        tot = self.time or 1.0
        out = {"compute": 0.0, "memory": 0.0, "transfer": 0.0}
        for c in self.costs:
            if c.bound in ("compute", "memory"):
                out[c.bound] += c.time / tot
            out["transfer"] += c.transfer / tot
        return out


class CostModel(Protocol):
    name: str

    def run(self, trace: Trace) -> CostReport: ...


# ── roofline ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Roofline:
    name: str
    flops_rate: float          # achievable FLOP/s
    byte_rate: float           # achievable bytes/s
    op_overhead: float = 0.0   # seconds per non-free operator (kernel launch); 0 = pure roofline
    memory: str = "unfused"    # "unfused" or "fused" (ideal fusion bound)

    @classmethod
    def from_device(cls, dev, n_devices: int = 1, **kw) -> "Roofline":
        """From a ``disagg_sim.hardware.Accelerator``: peak x efficiency x devices, as its CostModel does."""
        return cls(dev.name, dev.peak_flops * dev.flops_eff * n_devices, dev.mem_bw * dev.bw_eff * n_devices, **kw)

    def op_cost(self, op: Op) -> OpCost:
        if op.category in ("view", "unknown"):
            return OpCost(0.0, "free", "-")
        nbytes = op.bytes if (self.memory == "unfused" or op.category in MEMORY_OPS) else 0
        tc, tm = op.flops / self.flops_rate, nbytes / self.byte_rate
        return OpCost(max(tc, tm) + self.op_overhead, "compute" if tc >= tm else "memory", "accel")

    def run(self, trace: Trace) -> CostReport:
        costs = [self.op_cost(o) for o in trace.ops]
        tc = sum(o.flops for o in trace.ops) / self.flops_rate
        tm = sum(o.bytes for o in trace.ops if self.memory == "unfused" or o.category in MEMORY_OPS) / self.byte_rate
        return CostReport(self.name, trace, costs, {"whole_trace_roofline_s": max(tc, tm), "compute_s": tc,
                                                    "memory_s": tm})


# ── offload ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Offload:
    """An accelerator for some categories, a host for the rest, a link between them.

    Weights are resident on both sides. Tensors that exist before the trace starts (the model
    input, buffers, a KV cache from earlier steps) start on the accelerator. A tensor is moved
    the first time an operator on the other side reads it (link latency + bytes / bandwidth)
    and is then resident on both. Views and unknown operators run wherever their input lives,
    at no cost.
    """

    name: str
    accel: Roofline
    host: Roofline
    link_bw: float
    link_latency: float
    supported: frozenset[str]

    def run(self, trace: Trace) -> CostReport:
        where: dict[str, set[str]] = {}
        costs = []
        for o in trace.ops:
            if o.category in ("view", "unknown"):
                res = where.get(o.inputs[0].tid, {"accel"}) if o.inputs else {"accel"}
                for t in o.outputs:
                    where[t.tid] = set(res)
                costs.append(OpCost(0.0, "free", "-"))
                continue
            side = "accel" if o.category in self.supported else "host"
            moved = 0.0
            for t in o.inputs:
                if t.param:
                    continue
                res = where.setdefault(t.tid, {"accel"})
                if side not in res:
                    moved += self.link_latency + t.nbytes / self.link_bw
                    res.add(side)
            base = (self.accel if side == "accel" else self.host).op_cost(o)
            for t in o.outputs:
                where[t.tid] = {side}
            costs.append(OpCost(base.time, base.bound, side, moved))
        return CostReport(self.name, trace, costs)


# ── presets ──────────────────────────────────────────────────────────────
GB, TB = 1e9, 1e12
# Illustrative host: a server CPU with a matrix extension, after derating.
HOST_CPU = Roofline("Host CPU (illustrative)", flops_rate=2 * TB, byte_rate=200 * GB)


def device(name: str, n_devices: int = 1, **kw) -> Roofline:
    """``h100``, ``a100`` or ``optical`` from Disaggregated_Inference_Sim's ``ACCELERATORS``."""
    from disagg_sim.hardware import ACCELERATORS

    return Roofline.from_device(ACCELERATORS[name], n_devices, **kw)


def matmul_engine(name: str = "optical", supported: Iterable[str] = ("matmul",), **kw) -> Offload:
    """A device that runs only ``supported`` categories, with the rest on HOST_CPU over PCIe Gen5 x16."""
    from disagg_sim.hardware import LINKS

    link = LINKS["pcie5"]
    return Offload(f"{name} ({'+'.join(supported)} only) + host", device(name, **kw), HOST_CPU, link.bandwidth,
                   link.latency, frozenset(supported))
