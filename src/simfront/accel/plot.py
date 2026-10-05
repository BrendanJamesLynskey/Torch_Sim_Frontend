"""Timeline outputs: a PNG Gantt chart (matplotlib) and a Chrome trace for Perfetto.

The PNG has one lane per engine (load DMA, compute array, vector unit, in-transit stage,
store DMA), coloured by operator category, above two step plots: on-chip buffer occupancy
and the depth of the ready FIFO (tiles loaded and waiting for compute). The Chrome trace
holds the same intervals and opens at https://ui.perfetto.dev.
"""

from __future__ import annotations

import json
from pathlib import Path

from .lower import Program
from .metrics import Report
from .sim import Timings

COLOURS = {"matmul": "#3b6ea8", "attention": "#7a4fa3", "elementwise": "#d08c2e", "softmax": "#c0504d",
           "norm": "#5a9a54", "reduction": "#8c6d46", "copy": "#8a8a8a", "creation": "#b5b5b5",
           "gather": "#4aa0a0", "ntt": "#b8487a"}


def _lanes(prog: Program, tm: Timings):
    scale = 1.0 / prog.config.clock_hz if prog.config.quantise else 1.0
    lanes: dict[str, list[tuple[float, float, str]]] = {"load DMA": [], "compute array": [], "vector unit": [],
                                                         "in-transit": [], "store DMA": []}
    for i, t in enumerate(prog.tiles):
        cat = prog.ops[t.op].category
        if tm.load_end[i] > tm.alloc[i]:
            lanes["in-transit" if t.unit == "transit" else "load DMA"].append(
                (tm.alloc[i] * scale, (tm.load_end[i] - tm.alloc[i]) * scale, cat))
        if tm.comp_end[i] > tm.comp_start[i]:
            lanes["compute array" if t.unit == "array" else "vector unit"].append(
                (tm.comp_start[i] * scale, (tm.comp_end[i] - tm.comp_start[i]) * scale, cat))
        if t.stores and tm.store_end[i] > tm.store_start[i]:
            lanes["store DMA"].append((tm.store_start[i] * scale, (tm.store_end[i] - tm.store_start[i]) * scale, cat))
    return {k: v for k, v in lanes.items() if v}


def timeline_png(prog: Program, tm: Timings, rep: Report, path: str | Path, title: str | None = None,
                 window: tuple[float, float] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lanes = _lanes(prog, tm)
    names = list(lanes)[::-1]
    fig, (ax, ax_b, ax_q) = plt.subplots(3, 1, figsize=(11, 2.2 + 0.45 * len(names) + 2.6), sharex=True,
                                         gridspec_kw={"height_ratios": [0.5 * len(names) + 0.6, 1.2, 1.0]})
    unit = 1e6 if rep.makespan < 1e-3 else 1e3
    uname = "µs" if unit == 1e6 else "ms"
    used = set()
    for y, name in enumerate(names):
        for cat in sorted({c for _, _, c in lanes[name]}):
            bars = [(s * unit, d * unit) for s, d, c in lanes[name] if c == cat]
            ax.broken_barh(bars, (y - 0.38, 0.76), facecolors=COLOURS.get(cat, "#444"), linewidth=0)
            used.add(cat)
    ax.set_yticks(range(len(names)), names)
    ax.set_title(title or f"{rep.model} on {rep.config}: {rep.makespan * 1e3:,.3f} ms")
    ax.legend(handles=[Patch(color=COLOURS.get(c, "#444"), label=c) for c in sorted(used)], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=min(6, len(used)), fontsize=8, frameon=True, facecolor="white")
    xs = [t * unit for t, _ in rep.buffer_steps]
    ys = [v / prog.config.buffer_bytes * 100 for _, v in rep.buffer_steps]
    ax_b.step(xs, ys, where="post", color="#3b6ea8", linewidth=0.9)
    ax_b.set_ylabel("buffer\nused (%)")
    ax_b.set_ylim(0, 105)
    xs = [t * unit for t, _ in rep.ready_steps]
    ys = [v for _, v in rep.ready_steps]
    ax_q.step(xs, ys, where="post", color="#c0504d", linewidth=0.9)
    ax_q.set_ylabel("ready FIFO\n(tiles)")
    ax_q.set_xlabel(f"time ({uname})")
    if window:
        ax.set_xlim(window[0] * unit, window[1] * unit)
    for a in (ax, ax_b, ax_q):
        a.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def chrome_trace(prog: Program, tm: Timings, path: str | Path) -> Path:
    """Chrome trace-event JSON (``ph: "X"`` complete events in µs), one thread per lane."""
    lanes = _lanes(prog, tm)
    tids = {name: i + 1 for i, name in enumerate(lanes)}
    ev = [{"ph": "M", "name": "thread_name", "pid": 1, "tid": tid, "args": {"name": name}}
          for name, tid in tids.items()]
    ev.append({"ph": "M", "name": "process_name", "pid": 1, "args": {"name": f"{prog.model} on {prog.config.name}"}})
    for name, items in lanes.items():
        for s, d, cat in items:
            ev.append({"ph": "X", "name": cat, "cat": cat, "pid": 1, "tid": tids[name], "ts": s * 1e6, "dur": d * 1e6})
    path = Path(path)
    path.write_text(json.dumps({"traceEvents": ev, "displayTimeUnit": "ns"}))
    return path
