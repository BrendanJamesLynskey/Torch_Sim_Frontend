"""The same accelerator, cycle-stepped: the RTL-style twin of the event-driven model.

An RTL simulator (or a cycle-based architectural model) advances a clock and evaluates
every block on every edge, whether or not anything changes. This twin does the same to
the tile pipeline: on every cycle each engine (load DMA, compute, store DMA) is a small
state machine that finishes its current tile if its countdown has expired and starts the
next one if it can. The event-driven model (:mod:`simfront.accel.sim`) instead jumps from
one state change to the next.

Run on a program lowered with ``quantise=True`` (every duration a whole number of cycles),
the two produce **identical per-tile timings** (``tests/test_accel_cycle.py``). What differs
is the cost: the event-driven model's work grows with the number of state changes, the
cycle-stepped model's with the number of cycles, busy or idle. ``examples/accel_results.py``
measures both.

Within one cycle the order is: retire everything that finishes now (which frees buffer
space and marks operators done), then start everything that can start; repeated until
nothing changes, so zero-length steps chain within a cycle as they do at one SimPy instant.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from .lower import Program
from .sim import Timings


@dataclass
class CycleStats:
    cycles: int
    evaluations: int        # engine evaluations: 3 per cycle, plus the extra passes that chain zero-length steps
    wall_s: float


def simulate_cycles(prog: Program, max_cycles: int = 50_000_000) -> tuple[Timings, CycleStats]:
    cfg = prog.config
    if not cfg.quantise:
        raise ValueError("the cycle-stepped twin needs a program lowered with quantise=True")
    tiles = prog.tiles
    n = len(tiles)
    tm = Timings.empty(n, "cycle")
    first = {o.first_tile: o for o in prog.ops}
    op_done = [None] * len(prog.ops)                 # cycle the op's last store finished
    level = 0
    ready: deque[int] = deque()
    done: deque[int] = deque()
    nxt = 0                                          # next tile for the load DMA
    load = comp = store = None                       # (tile, finish cycle) of the current job
    load_free_at = 0                                 # cycle the load DMA became free (issue time)
    issued = dep_ok = False
    stored = 0
    n_stored = sum(1 for t in tiles if t.stores)
    evals = 0
    t0 = time.perf_counter()
    cyc = 0
    while stored < n_stored or nxt < n or comp is not None:
        changed = True
        while changed:
            changed = False
            evals += 3
            # ── retire ────────────────────────────────────────────────
            if load is not None and load[1] <= cyc:
                i = load[0]
                tm.load_end[i] = cyc
                ready.append(i)
                load = None
                load_free_at = cyc
                nxt += 1
                issued = dep_ok = False
                changed = True
            if comp is not None and comp[1] <= cyc:
                i = comp[0]
                t = tiles[i]
                tm.comp_end[i] = cyc
                if t.stores:
                    level -= t.alloc_in
                    done.append(i)
                else:
                    level -= t.alloc
                    tm.store_start[i] = tm.store_end[i] = cyc
                comp = None
                changed = True
            if store is not None and store[1] <= cyc:
                i = store[0]
                t = tiles[i]
                tm.store_end[i] = cyc
                level -= t.alloc_out
                if t.last:
                    op_done[t.op] = cyc
                stored += 1
                store = None
                changed = True
            # ── start ─────────────────────────────────────────────────
            if load is None and nxt < n:
                i = nxt
                t = tiles[i]
                if not issued:
                    tm.issue[i] = load_free_at
                    issued = True
                if not dep_ok:
                    o = first.get(i)
                    if o is None or all(op_done[d] is not None for d in o.deps):
                        tm.dep[i] = cyc
                        dep_ok = True
                if dep_ok and level + t.alloc <= cfg.buffer_bytes:
                    level += t.alloc
                    tm.alloc[i] = cyc
                    load = (i, cyc + int(t.load_dur))
                    changed = True
            if comp is None and ready:
                i = ready.popleft()
                tm.comp_start[i] = cyc
                comp = (i, cyc + int(tiles[i].comp_dur))
                changed = True
            if store is None and done:
                i = done.popleft()
                tm.store_start[i] = cyc
                store = (i, cyc + int(tiles[i].store_dur))
                changed = True
        if stored >= n_stored and nxt >= n and comp is None:
            break
        cyc += 1
        if cyc > max_cycles:
            raise RuntimeError(f"no progress after {max_cycles} cycles")
    for i in range(n):                               # float, like the event-driven model's times
        for f in ("issue", "dep", "alloc", "load_end", "comp_start", "comp_end", "store_start", "store_end"):
            getattr(tm, f)[i] = float(getattr(tm, f)[i])
    return tm, CycleStats(cyc, evals, time.perf_counter() - t0)
