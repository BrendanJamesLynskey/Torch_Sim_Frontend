"""The operator trace: what every front end produces and every cost model consumes.

A trace is a list of operators in execution order. Each operator names the tensors it
reads and writes (by id, so a cost model can follow data between devices), with
their shapes and element sizes, and carries the FLOPs and bytes its cost rule gave it.
Tensors that are model parameters are flagged, so weight traffic can be told apart
from activation traffic.

The format is the same whichever front end produced it (dispatch trace, torch.export,
torch.compile or ONNX), which is what makes the front ends comparable.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TensorMeta:
    """A tensor as the cost model sees it: an id, a shape, an element size, and whether it is a weight."""

    tid: str
    shape: tuple[int, ...]
    itemsize: int
    param: bool = False

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize


@dataclass
class Op:
    """One operator call. ``name`` is canonical: ``aten.mm`` or ``onnx.MatMul``."""

    name: str
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    attrs: dict = field(default_factory=dict)
    category: str = "unknown"
    flops: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    weight_bytes: int = 0      # the part of bytes_read that is parameters

    @property
    def bytes(self) -> int:
        return self.bytes_read + self.bytes_written


@dataclass
class Trace:
    """An operator trace plus where it came from."""

    model: str
    route: str                 # "dispatch", "export", "compile" or "onnx"
    ops: list[Op]
    workload: dict = field(default_factory=dict)
    capture_s: float = 0.0     # wall-clock time the front end took

    # ── summaries ──────────────────────────────────────────────────────
    @property
    def flops(self) -> int:
        return sum(o.flops for o in self.ops)

    @property
    def bytes(self) -> int:
        return sum(o.bytes for o in self.ops)

    def flops_of(self, *categories: str) -> int:
        return sum(o.flops for o in self.ops if o.category in categories)

    @property
    def weight_bytes(self) -> int:
        return sum(o.weight_bytes for o in self.ops)

    def counts(self) -> Counter:
        return Counter(o.name for o in self.ops)

    def by_category(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for o in self.ops:
            c = out.setdefault(o.category, {"ops": 0, "flops": 0, "bytes": 0})
            c["ops"] += 1
            c["flops"] += o.flops
            c["bytes"] += o.bytes
        return out

    def unknown(self) -> Counter:
        """Operators with no cost rule: they are costed at zero, so they must be visible."""
        return Counter(o.name for o in self.ops if o.category == "unknown")

    # ── files ──────────────────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps({"format": "simfront-trace/1", "model": self.model, "route": self.route,
                           "workload": self.workload, "capture_s": self.capture_s,
                           "ops": [asdict(o) for o in self.ops]})

    @classmethod
    def from_json(cls, text: str) -> "Trace":
        d = json.loads(text)
        if d.get("format") != "simfront-trace/1":
            raise ValueError("not a simfront-trace/1 file")
        ops = []
        for o in d["ops"]:
            o["inputs"] = [TensorMeta(t["tid"], tuple(t["shape"]), t["itemsize"], t["param"]) for t in o["inputs"]]
            o["outputs"] = [TensorMeta(t["tid"], tuple(t["shape"]), t["itemsize"], t["param"]) for t in o["outputs"]]
            ops.append(Op(**o))
        return cls(d["model"], d["route"], ops, d["workload"], d["capture_s"])
