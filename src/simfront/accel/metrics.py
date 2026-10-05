"""Metrics from per-tile timings: latency, utilisation per component, stalls and the hot-spot.

Everything here is a function of a :class:`~simfront.accel.lower.Program` and the
:class:`~simfront.accel.sim.Timings` of a run, so every engine (SimPy, the Python
recurrence, the C++ module, the cycle-stepped twin) gets the same report.

**Stall attribution.** The compute units issue in order, so the run's time splits exactly
into time computing and gaps between one tile's compute and the next. A gap means the next
tile's operands were not loaded yet; the gap is attributed by what its load was waiting for,
walking back from the load's end:

* ``load bandwidth``: the load was transferring, or the load DMA was still busy with
  earlier tiles (the memory path is the bottleneck);
* ``buffer full``: the load waited for buffer space (back-pressure from compute or the store
  path);
* ``dependency``: the load waited for an earlier operator's results to be stored, because
  every operator reads its inputs from off-chip memory (the pipeline drains at operator
  boundaries);
* ``store tail``: after the last compute, the time left to store the results.

``compute`` + the four stall classes = the makespan, for every run (a tested invariant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .lower import Program
from .sim import Timings

STALLS = ("load bandwidth", "buffer full", "dependency", "store tail")


def _steps(events: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Cumulative level after each time in ``events`` [(time, delta)], one point per distinct time."""
    events.sort(key=lambda e: e[0])
    out: list[tuple[float, float]] = []
    level = 0.0
    for t, d in events:
        level += d
        if out and out[-1][0] == t:
            out[-1] = (t, level)
        else:
            out.append((t, level))
    return out


def _time_avg(steps: list[tuple[float, float]], end: float) -> float:
    if not steps or end <= 0:
        return 0.0
    area = 0.0
    for (t0, v), (t1, _) in zip(steps, steps[1:] + [(end, 0.0)], strict=True):
        area += v * (t1 - t0)
    return area / end


def histogram(values: list[float], bins: int = 10, log: bool = True) -> list[tuple[float, float, int]]:
    """(low, high, count) bins; logarithmic bins by default, as latencies span decades."""
    vals = [v for v in values if v > 0]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return [(lo, hi, len(vals))]
    if log:
        a, b = math.log10(lo), math.log10(hi)
        edges = [10 ** (a + (b - a) * i / bins) for i in range(bins + 1)]
    else:
        edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in vals:
        j = bins - 1
        for k in range(bins):
            if v < edges[k + 1]:
                j = k
                break
        counts[j] += 1
    return [(edges[i], edges[i + 1], counts[i]) for i in range(bins)]


@dataclass
class Report:
    model: str
    config: str
    engine: str
    makespan: float                                  # seconds
    components: dict[str, dict]                      # name -> {busy, bytes, bw_util, ...}
    stalls: dict[str, float]                         # seconds, including "compute"
    hotspot: str
    hotspot_reason: str
    ops: list[dict]                                  # per operator: name, category, unit, span, bound, ...
    buffer_steps: list[tuple[float, float]]
    ready_steps: list[tuple[float, float]]
    unknown: dict[str, int] = field(default_factory=dict)
    tiles: int = 0
    traffic: dict[str, int] = field(default_factory=dict)

    def op_latency_histogram(self, bins: int = 10):
        return histogram([o["span"] for o in self.ops], bins)

    def to_markdown(self, top: int = 8) -> str:
        ms = self.makespan * 1e3
        lines = [f"**{self.model} on {self.config}** ({self.engine}): {ms:,.4f} ms, {self.tiles:,} tiles, "
                 f"{len(self.ops):,} operators", "",
                 "| Component | Busy | Bytes | Bandwidth used |", "|---|---|---|---|"]
        for name, c in self.components.items():
            bw = f"{c['bw_util']:.1%}" if c.get("bw_util") is not None else "—"
            by = f"{c['bytes'] / 1e6:,.2f} MB" if c.get("bytes") is not None else "—"
            extra = f" (PE efficiency {c['pe_eff']:.1%})" if c.get("pe_eff") is not None else ""
            if c.get("peak") is not None:
                extra = f" mean occupancy (peak {c['peak']:.1%})"
            lines.append(f"| {name} | {c['busy']:.1%}{extra} | {by} | {bw} |")
        lines += ["", "| Where the compute units' time went | Share |", "|---|---|"]
        for k, v in self.stalls.items():
            lines.append(f"| {k} | {v / self.makespan:.1%} |" if self.makespan else f"| {k} | — |")
        lines += ["", f"**Hot-spot: {self.hotspot}.** {self.hotspot_reason}", "",
                  "| Top operators by time | Category | Unit | Time (µs) | Share | Bound |",
                  "|---|---|---|---|---|---|"]
        for o in sorted(self.ops, key=lambda o: -o["span"])[:top]:
            lines.append(f"| {o['name']} | {o['category']} | {o['unit']} | {o['span'] * 1e6:,.2f} | "
                         f"{o['span'] / self.makespan:.1%} | {o['bound']} |")
        if self.unknown:
            lines += ["", "Operators with no cost rule (not simulated): "
                      + ", ".join(f"{k} x{v}" for k, v in sorted(self.unknown.items()))]
        return "\n".join(lines)


def report(prog: Program, tm: Timings) -> Report:
    cfg = prog.config
    scale = 1.0 / cfg.clock_hz if cfg.quantise else 1.0       # report in seconds
    tiles = prog.tiles
    n = len(tiles)
    span = tm.makespan
    mk = span * scale

    def busy(sel, start, end):
        return sum((end[i] - start[i]) for i in range(n) if sel(i)) * scale

    load_busy = busy(lambda i: True, tm.alloc, tm.load_end)
    store_busy = busy(lambda i: tiles[i].stores, tm.store_start, tm.store_end)
    unit_busy = {u: busy(lambda i, u=u: tiles[i].unit == u, tm.comp_start, tm.comp_end) for u in ("array", "vector")}
    transit_busy = busy(lambda i: tiles[i].unit == "transit", tm.alloc, tm.load_end)
    rd = sum(t.load_bytes for t in tiles)
    wr = sum(t.store_bytes for t in tiles)
    macs = sum(t.macs for t in tiles)
    comps: dict[str, dict] = {
        "off-chip memory": {"busy": (load_busy + store_busy) / cfg.dram_channels / mk if mk else 0.0,
                            "bytes": rd + wr, "bw_util": (rd + wr) / (cfg.dram_bw * mk) if mk else 0.0},
        "NoC read": {"busy": load_busy / mk if mk else 0.0, "bytes": rd,
                     "bw_util": rd / (cfg.noc_bw * mk) if mk else 0.0},
        "NoC write": {"busy": store_busy / mk if mk else 0.0, "bytes": wr,
                      "bw_util": wr / (cfg.noc_bw * mk) if mk else 0.0},
        "load DMA": {"busy": load_busy / mk if mk else 0.0, "bytes": rd},
        "store DMA": {"busy": store_busy / mk if mk else 0.0, "bytes": wr},
        "compute array": {"busy": unit_busy["array"] / mk if mk else 0.0, "bytes": None,
                          "pe_eff": macs / (cfg.peak_macs * unit_busy["array"]) if unit_busy["array"] else None},
        "vector unit": {"busy": unit_busy["vector"] / mk if mk else 0.0, "bytes": None},
    }
    if transit_busy:
        comps["in-transit stage"] = {"busy": transit_busy / mk if mk else 0.0, "bytes": None}

    # ── occupancy and FIFO depth over time ─────────────────────────────
    buf_ev: list[tuple[float, float]] = []
    rdy_ev: list[tuple[float, float]] = []
    for i, t in enumerate(tiles):
        if t.alloc:
            buf_ev.append((tm.alloc[i] * scale, t.alloc))
            if t.stores:
                if t.alloc_in:
                    buf_ev.append((tm.comp_end[i] * scale, -t.alloc_in))
                if t.alloc_out:
                    buf_ev.append((tm.store_end[i] * scale, -t.alloc_out))
            else:
                buf_ev.append((tm.comp_end[i] * scale, -t.alloc))
        rdy_ev.append((tm.load_end[i] * scale, 1))
        rdy_ev.append((tm.comp_start[i] * scale, -1))
    buf_steps = _steps(buf_ev)
    rdy_steps = _steps(rdy_ev)
    peak = max((v for _, v in buf_steps), default=0.0)
    comps["on-chip buffer"] = {"busy": _time_avg(buf_steps, mk) / cfg.buffer_bytes, "bytes": None,
                               "peak": peak / cfg.buffer_bytes}

    # ── stalls: walk each compute gap back along its load ───────────────
    st = dict.fromkeys(("compute", *STALLS), 0.0)
    transit_wait = 0.0                                    # the part of "load bandwidth" spent in the transit stage
    prev = 0.0
    for i in range(n):
        st["compute"] += tm.comp_end[i] - tm.comp_start[i]
        gap_lo, gap_hi = prev, tm.comp_start[i]
        if gap_hi > gap_lo:
            segs = (("load bandwidth", tm.alloc[i], tm.load_end[i]), ("buffer full", tm.dep[i], tm.alloc[i]),
                    ("dependency", tm.issue[i], tm.dep[i]), ("load bandwidth", -math.inf, tm.issue[i]))
            for k, (name, a, b) in enumerate(segs):
                d = max(0.0, min(b, gap_hi) - max(a, gap_lo))
                st[name] += d
                if k == 0 and tiles[i].unit == "transit":
                    transit_wait += d
        prev = tm.comp_end[i]
    st["store tail"] += span - prev
    st = {k: v * scale for k, v in st.items()}
    transit_wait *= scale

    # ── per operator ────────────────────────────────────────────────────
    ops = []
    for o in prog.ops:
        lo, hi = o.first_tile, o.first_tile + o.n_tiles
        start = min(tm.alloc[lo:hi]) * scale
        end = max(tm.store_end[lo:hi]) * scale
        ld = sum(tiles[i].load_dur + tiles[i].store_dur for i in range(lo, hi)) * scale
        cp = sum(tiles[i].comp_dur for i in range(lo, hi)) * scale
        ops.append({"name": f"{o.name} #{o.index}", "category": o.category, "unit": o.unit, "span": end - start,
                    "start": start,
                    "end": end, "bound": "compute" if cp >= ld else "memory", "tiles": o.n_tiles, "flops": o.flops,
                    "bytes": o.bytes, "tiled_bytes": o.tiled_bytes})

    # ── hot-spot ────────────────────────────────────────────────────────
    stall_k = max(STALLS, key=lambda k: st[k])
    if st["compute"] >= st[stall_k]:
        u = max(("array", "vector"), key=lambda u: unit_busy[u])
        hot = "compute array" if u == "array" else "vector unit"
        why = (f"The compute units are busy {st['compute'] / mk:.1%} of the run, most of it on the {hot}; "
               "a faster unit (or more of them) shortens the run, more bandwidth does not.") if mk else ""
    elif stall_k == "load bandwidth" and transit_wait > 0.5 * st[stall_k]:
        hot = "in-transit stage"
        why = (f"Compute waits {st[stall_k] / mk:.1%} of the run for operand loads, {transit_wait / mk:.1%} of it in "
               f"loads slowed to the in-transit stage's {cfg.transit_ops_per_byte} operations per byte.")
    elif stall_k == "load bandwidth":
        hot = "off-chip memory" if cfg.path_limit == "dram" else "NoC read"
        why = (f"Compute waits {st[stall_k] / mk:.1%} of the run for operand loads; one transfer runs at "
               f"{cfg.path_bw / 1e9:,.1f} GB/s, set by the {'memory channel' if hot == 'off-chip memory' else 'NoC'}.")
    elif stall_k == "dependency":
        hot = "operator boundaries (results round-trip through off-chip memory)"
        why = (f"Compute waits {st[stall_k] / mk:.1%} of the run for an earlier operator's results to be stored and "
               "reloaded; keeping results on chip (fusion) removes this.")
    elif stall_k == "buffer full":
        hot = "on-chip buffer"
        why = f"Loads wait {st[stall_k] / mk:.1%} of the run for buffer space (back-pressure)."
    else:
        hot = "store path"
        why = f"{st[stall_k] / mk:.1%} of the run is the final stores after compute ends."
    traffic = {"op_bytes": sum(o.bytes for o in prog.ops), "tiled_bytes": rd + wr}
    return Report(prog.model, cfg.name, tm.engine, mk, comps, st, hot, why, ops, buf_steps, rdy_steps,
                  dict(prog.unknown), n, traffic)
