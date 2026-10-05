"""The fast path: the same tile pipeline as a recurrence instead of an event loop.

When the load and store DMA engines never compete for a memory channel
(``dram_channels >= 2``; the NoC has separate read and write networks), the event-driven
model has no arbitration left to decide, and every tile's times follow from earlier
tiles' times alone::

    issue_i     = load_end_{i-1}
    dep_i       = max(issue_i, store_end of every operator tile i's operator reads)
    alloc_i     = earliest t >= dep_i at which the buffer has room, given the frees so far
    load_end_i  = alloc_i + load_dur_i
    comp_start  = max(load_end_i, comp_end_{i-1});         comp_end  = comp_start + comp_dur_i
    store_start = max(comp_end_i, store_end_{prev stored}); store_end = store_start + store_dur_i

The frees (operands at ``comp_end``, results at ``store_end``) go into a min-heap; finding
``alloc_i`` pops frees until the tile fits. The arithmetic is the same additions and
maxima, in the same order, as the SimPy model, so the results are **bit-identical**
(``tests/test_accel_fastpath.py`` checks every field of every tile).

Two implementations: :func:`run_py` below, and the C++20 module ``_fastpath``
(``src/simfront/accel/_fastpath.cpp``, built with pybind11), which ports it line for line.
"""

from __future__ import annotations

import heapq

from .lower import Program
from .sim import Timings

try:                                                     # the compiled module is optional at import time
    from . import _fastpath as _cpp  # type: ignore[attr-defined]
except ImportError:                                      # pragma: no cover - exercised when the build is absent
    _cpp = None


def available() -> bool:
    return _cpp is not None


def check(prog: Program) -> None:
    if prog.config.dram_channels < 2:
        raise ValueError("the fast path needs dram_channels >= 2 (load and store would contend for one channel); "
                         "use simulate()")
    for t in prog.tiles:
        if t.alloc > prog.config.buffer_bytes:
            raise ValueError(f"tile of op {t.op} needs {t.alloc} bytes, buffer is {prog.config.buffer_bytes}")


def run_py(prog: Program) -> Timings:
    check(prog)
    a = prog.arrays()
    return _recurrence(a, Timings.empty(prog.n_tiles, "python"))


def _recurrence(a: dict, tm: Timings) -> Timings:
    cap = a["capacity"]
    op_done = [0.0] * a["n_ops"]
    frees: list[tuple[float, int, int]] = []             # (time, sequence, bytes)
    seq = 0
    level = 0
    load_end = comp_end = store_end = 0.0
    for i in range(len(a["op"])):
        t = load_end
        tm.issue[i] = t
        if a["first"][i]:
            for d in a["deps"][i]:
                if op_done[d] > t:
                    t = op_done[d]
        tm.dep[i] = t
        need = a["alloc_in"][i] + a["alloc_out"][i]
        while frees and frees[0][0] <= t:
            level -= heapq.heappop(frees)[2]
        while level + need > cap:
            ft, _, b = heapq.heappop(frees)
            level -= b
            if ft > t:
                t = ft
        level += need
        tm.alloc[i] = t
        if a["load_dur"][i] > 0:
            t = t + a["load_dur"][i]
        load_end = t
        tm.load_end[i] = t
        c = load_end if load_end > comp_end else comp_end
        tm.comp_start[i] = c
        if a["comp_dur"][i] > 0:
            c = c + a["comp_dur"][i]
        comp_end = c
        tm.comp_end[i] = c
        if a["stores"][i]:
            if a["alloc_in"][i]:
                heapq.heappush(frees, (c, seq, a["alloc_in"][i]))
                seq += 1
            s = c if c > store_end else store_end
            tm.store_start[i] = s
            if a["store_dur"][i] > 0:
                s = s + a["store_dur"][i]
            store_end = s
            tm.store_end[i] = s
            if a["alloc_out"][i]:
                heapq.heappush(frees, (s, seq, a["alloc_out"][i]))
                seq += 1
            if a["last"][i]:
                op_done[a["op"][i]] = s
        else:
            if need:
                heapq.heappush(frees, (c, seq, need))
                seq += 1
            tm.store_start[i] = tm.store_end[i] = c
    return tm


def run_cpp(prog: Program) -> Timings:
    if _cpp is None:
        raise RuntimeError("simfront.accel._fastpath is not built (pip install -e . with a C++20 compiler)")
    check(prog)
    a = prog.arrays()
    cols = _cpp.run(a["op"], a["first"], a["last"], a["stores"], a["alloc_in"], a["alloc_out"], a["load_dur"],
                    a["comp_dur"], a["store_dur"], a["deps"], a["n_ops"], a["capacity"])
    return Timings(*cols, engine="c++")


def run(prog: Program, engine: str = "auto") -> Timings:
    """``engine``: ``"c++"``, ``"python"`` or ``"auto"`` (C++ when built)."""
    if engine == "c++" or (engine == "auto" and _cpp is not None):
        return run_cpp(prog)
    return run_py(prog)
