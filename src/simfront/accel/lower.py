"""Lower an operator trace onto the accelerator: operators become tiles, tiles get durations.

A **tile** is the unit the hardware moves and computes: the load DMA brings its operands
into the on-chip buffer, a compute unit processes it, and the store DMA writes its results
back. Lowering decides, for every operator in the trace:

* **which unit runs it.** Matmuls, convolutions and fused attention go to the compute
  array; elementwise, softmax, norm, reduction, copy, creation and gather operators go to
  the vector unit; NTTs go to the vector unit, or to the in-transit stage when one is
  configured. Views are free and produce no tiles. Operators with no cost rule (category
  ``unknown``) produce no tiles and are counted in ``Program.unknown``.
* **how it is tiled.** A GEMM of shape (batch, M, K, N) is cut into blocks whose operands
  and result fit in ``tile_budget`` bytes, halving the largest block dimension until they
  fit. Every tile loads both operand blocks (no reuse across tiles: the simplest dataflow,
  and the one that shows the cost of tiling as extra memory traffic). A tile that is not
  the last slice of K keeps its partial sums on chip and stores nothing. Other operators
  are cut into equal slices of their bytes.
* **what it waits for.** Every operator reads its activations from off-chip memory, so the
  first load of an operator waits until every operator that produced one of its inputs has
  been stored (a read-after-write dependency through memory). Weights are always ready.

All durations are computed here, once, by :class:`~simfront.accel.hw.AccelConfig`. The
event-driven model, its compiled fast path and the cycle-stepped twin all run the same
:class:`Program`, which is what makes them comparable value for value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..rules import aten_packet
from ..trace import Op, Trace
from .hw import AccelConfig

ARRAY_CATS = {"matmul", "attention"}
VECTOR_CATS = {"elementwise", "softmax", "norm", "reduction", "copy", "creation", "gather", "ntt"}
UNITS = ("array", "vector", "transit")


@dataclass
class Tile:
    op: int                 # index into Program.ops
    unit: str               # "array", "vector" or "transit" (computed in the read path)
    load_bytes: int
    store_bytes: int
    alloc_in: int           # buffer bytes freed when compute ends
    alloc_out: int          # buffer bytes freed when the store ends (or when compute ends, if not stored)
    load_dur: float
    comp_dur: float
    store_dur: float
    stores: bool            # goes through the store DMA (it writes results, or it is the op's last tile)
    last: bool              # last tile of its operator
    macs: int = 0           # useful multiply-accumulates (array tiles)
    work: int = 0           # vector elements or NTT butterflies

    @property
    def alloc(self) -> int:
        return self.alloc_in + self.alloc_out


@dataclass
class OpInfo:
    index: int              # index in the trace
    name: str
    category: str
    unit: str
    deps: list[int]         # Program.ops indices whose stores this operator's loads wait for
    first_tile: int
    n_tiles: int
    flops: int
    bytes: int              # the operator's own bytes (from its cost rule)
    tiled_bytes: int        # bytes the tiles actually move (re-fetches included)


@dataclass
class Program:
    config: AccelConfig
    model: str
    tiles: list[Tile]
    ops: list[OpInfo]
    unknown: dict[str, int] = field(default_factory=dict)

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    def arrays(self) -> dict[str, list]:
        """The program as flat columns, the form the compiled fast path takes."""
        t = self.tiles
        first = [False] * len(t)
        dep_lists = [[] for _ in t]
        for o in self.ops:
            first[o.first_tile] = True
            dep_lists[o.first_tile] = o.deps
        return {
            "op": [x.op for x in t], "first": first, "last": [x.last for x in t], "stores": [x.stores for x in t],
            "alloc_in": [x.alloc_in for x in t], "alloc_out": [x.alloc_out for x in t],
            "load_dur": [x.load_dur for x in t], "comp_dur": [x.comp_dur for x in t],
            "store_dur": [x.store_dur for x in t], "deps": dep_lists, "n_ops": len(self.ops),
            "capacity": self.config.buffer_bytes,
        }


# ── GEMM shapes ──────────────────────────────────────────────────────────
def gemm_dims(op: Op) -> list[tuple[int, int, int, int]]:
    """(batch, M, K, N) of each GEMM an array operator performs; [] if the shapes do not say."""
    ins = [t.shape for t in op.inputs]
    out = op.outputs[0].shape if op.outputs else ()
    kind = op.name.split(".", 1)[1] if "." in op.name else op.name
    kind = aten_packet("aten." + kind) if op.name.startswith("aten.") else kind
    try:
        if op.category == "attention":                              # QK^T then AV, per batch x head
            q, k, v = ins[:3]
            bh = math.prod(q[:-2])
            return [(bh, q[-2], q[-1], k[-2]), (bh, q[-2], k[-2], v[-1])]
        if kind in ("mm",):
            return [(1, ins[0][0], ins[0][1], ins[1][1])]
        if kind in ("addmm",):
            return [(1, ins[1][0], ins[1][1], ins[2][1])]
        if kind in ("bmm",):
            return [(ins[0][0], ins[0][1], ins[0][2], ins[1][2])]
        if kind in ("baddbmm",):
            return [(ins[1][0], ins[1][1], ins[1][2], ins[2][2])]
        if kind in ("matmul", "MatMul") and len(ins[0]) >= 2 and len(ins[1]) >= 2:
            m, k_, n = ins[0][-2], ins[0][-1], ins[1][-1]
            return [(math.prod(out) // (m * n), m, k_, n)]
        if kind == "linear":
            k_ = ins[0][-1]
            return [(1, math.prod(ins[0]) // k_, k_, ins[1][0])]
        if kind == "Gemm":
            ta = op.attrs.get("transA", 0)
            m, k_ = (ins[0][1], ins[0][0]) if ta else (ins[0][0], ins[0][1])
            return [(1, m, k_, out[-1])]
        if kind in ("convolution", "_convolution", "conv1d", "conv2d", "conv3d", "Conv"):
            x, w = ins[0], ins[1]                                   # x (B, C_in, ...), w (C_out, C_in/g, *k)
            groups = x[1] // w[1]
            return [(groups, out[0] * math.prod(out[2:]), math.prod(w[1:]), w[0] // groups)]
    except (IndexError, ZeroDivisionError, TypeError):
        return []
    return []


def _split_gemm(b: int, m: int, k: int, n: int, e_in: int, e_out: int, budget: int):
    """Block sizes (bb, bm, bk, bn) whose operands and result fit in ``budget`` bytes."""
    def size(bb, bm, bk, bn):
        return bb * (e_in * (bm * bk + bk * bn) + e_out * bm * bn)

    bm, bk, bn = m, k, n
    while size(1, bm, bk, bn) > budget:
        big = max(bm, bk, bn)
        if big == 1:
            raise ValueError(f"a 1x1x1 GEMM block does not fit in a {budget}-byte tile")
        if bm == big:
            bm = math.ceil(bm / 2)
        elif bn == big:
            bn = math.ceil(bn / 2)
        else:
            bk = math.ceil(bk / 2)
    bb = max(1, min(b, budget // size(1, bm, bk, bn))) if (bm, bk, bn) == (m, k, n) else 1
    return bb, bm, bk, bn


def _gemm_tiles(cfg: AccelConfig, op_i: int, gemms, e_in: int, e_out: int) -> list[Tile]:
    tiles: list[Tile] = []
    for b, m, k, n in gemms:
        bb, bm, bk, bn = _split_gemm(b, m, k, n, e_in, e_out, cfg.tile_budget)
        for b0 in range(0, b, bb):
            nb = min(bb, b - b0)
            for m0 in range(0, m, bm):
                mm = min(bm, m - m0)
                for n0 in range(0, n, bn):
                    nn = min(bn, n - n0)
                    for k0 in range(0, k, bk):
                        kk = min(bk, k - k0)
                        last_k = k0 + bk >= k
                        load = nb * e_in * (mm * kk + kk * nn)
                        out = nb * e_out * mm * nn
                        store = out if last_k else 0
                        tiles.append(Tile(op_i, "array", load, store, load, out, cfg.transfer_time(load),
                                          cfg.array_time(nb, mm, kk, nn), cfg.transfer_time(store), last_k, False,
                                          macs=nb * mm * kk * nn))
    return tiles


def _slice_tiles(cfg: AccelConfig, op_i: int, op: Op, unit: str) -> list[Tile]:
    """Equal slices of an operator's bytes, each within the tile budget."""
    rd, wr = op.bytes_read, op.bytes_written
    work = op.attrs.get("butterflies", 0) if op.category == "ntt" else max(op.flops, sum(t.numel for t in op.outputs))
    ops_total = op.flops if unit == "transit" else 0
    n = max(1, math.ceil((rd + wr) / cfg.tile_budget))
    while -(-rd // n) - (-wr // n) > cfg.tile_budget:            # integer slices: round up, then check
        n += 1
    tiles = []
    for i in range(n):
        load = rd * (i + 1) // n - rd * i // n
        store = wr * (i + 1) // n - wr * i // n
        w = work * (i + 1) // n - work * i // n
        o = ops_total * (i + 1) // n - ops_total * i // n
        comp = 0.0 if unit == "transit" else cfg.vector_time(w)
        tiles.append(Tile(op_i, unit, load, store, load, store, cfg.transfer_time(load, o), comp,
                          cfg.transfer_time(store), True, False, work=w))
    return tiles


def lower(trace: Trace, cfg: AccelConfig) -> Program:
    """Turn a trace into a tile program for ``cfg``."""
    producer: dict[str, int] = {}           # tensor id -> Program.ops index of the op that wrote it
    tiles: list[Tile] = []
    ops: list[OpInfo] = []
    unknown: dict[str, int] = {}
    for ti, op in enumerate(trace.ops):
        if op.category in ("view", "unknown"):
            if op.category == "unknown":
                unknown[op.name] = unknown.get(op.name, 0) + 1
            src = next((producer[t.tid] for t in op.inputs if t.tid in producer), None)
            for t in op.outputs:                                     # a view aliases its input's producer
                if src is not None:
                    producer[t.tid] = src
            continue
        if op.category in ARRAY_CATS:
            unit = "array"
        elif op.category in cfg.transit_categories and cfg.transit_ops_per_byte:
            unit = "transit"
        elif op.category in VECTOR_CATS:
            unit = "vector"
        else:
            raise ValueError(f"no unit for category {op.category!r} ({op.name})")
        idx = len(ops)
        new: list[Tile] = []
        if unit == "array":
            gemms = gemm_dims(op)
            e_in = op.inputs[0].itemsize if op.inputs else 2
            e_out = op.outputs[0].itemsize if op.outputs else e_in
            if gemms:
                new = _gemm_tiles(cfg, idx, gemms, e_in, e_out)
            else:                                                     # shapes unknown: slice by bytes, ideal array
                new = _slice_tiles(cfg, idx, op, "vector")
                for t in new:
                    t.unit = "array"
                    t.macs = op.flops // 2 // len(new)
                    t.comp_dur = cfg._time(math.ceil(t.macs / (cfg.array_rows * cfg.array_cols)) / cfg.clock_hz)
        else:
            new = _slice_tiles(cfg, idx, op, unit)
        new[-1].last = True
        new[-1].stores = True
        deps = sorted({producer[t.tid] for t in op.inputs if not t.param and t.tid in producer})
        ops.append(OpInfo(ti, op.name, op.category, unit, deps, len(tiles), len(new), op.flops, op.bytes,
                          sum(t.load_bytes + t.store_bytes for t in new)))
        tiles.extend(new)
        for t in op.outputs:
            producer[t.tid] = idx
    return Program(cfg, trace.model, tiles, ops, unknown)
