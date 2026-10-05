"""The accelerator as SimPy processes and resources.

=================  =======================================  ==============================================
component          SimPy primitive                          behaviour
=================  =======================================  ==============================================
off-chip memory    ``Resource(capacity=dram_channels)``     a transfer holds one channel for its duration
NoC read / write   two ``Resource(capacity=1)``             separate networks per direction, as on AXI
on-chip buffer     ``Container(capacity=buffer_bytes)``     the load DMA allocates a tile's operands and
                                                            result; compute frees the operands, the store
                                                            frees the result: **back-pressure** when full
ready / done FIFO  two ``Store()``                          loaded tiles waiting for compute; computed
                                                            tiles waiting for the store DMA
load DMA           process                                  in program order: wait for dependencies,
                                                            allocate, transfer, hand to compute
compute            process + ``Resource`` per unit          in-order, single issue: one tile at a time on
                                                            the array, the vector unit or (no time) in
                                                            transit
store DMA          process                                  write results back; the last tile of an
                                                            operator marks it done (``Event``)
=================  =======================================  ==============================================

The model records, for every tile, when the load DMA reached it, when its dependencies
were satisfied, when buffer space was allocated, and when its load, compute and store
started and ended. Every metric in :mod:`simfront.accel.metrics` is computed from these
timings, so the compiled fast path (which produces the same timings) gets the same report.
"""

from __future__ import annotations

from dataclasses import dataclass

import simpy

from .lower import Program

FIELDS = ("issue", "dep", "alloc", "load_end", "comp_start", "comp_end", "store_start", "store_end")


@dataclass
class Timings:
    """Per-tile event times, one list per field in ``FIELDS``; ``alloc`` is also the load start."""

    issue: list[float]
    dep: list[float]
    alloc: list[float]
    load_end: list[float]
    comp_start: list[float]
    comp_end: list[float]
    store_start: list[float]
    store_end: list[float]
    engine: str = "simpy"

    @classmethod
    def empty(cls, n: int, engine: str) -> "Timings":
        return cls(*([0.0] * n for _ in FIELDS), engine=engine)

    def as_tuple(self):
        return tuple(tuple(getattr(self, f)) for f in FIELDS)

    @property
    def makespan(self) -> float:
        return max(self.store_end + self.comp_end, default=0.0)


@dataclass
class SimStats:
    events: int
    wall_s: float


def simulate(prog: Program) -> tuple[Timings, SimStats]:
    """Run ``prog`` on the event-driven model; returns per-tile timings and the run's cost."""
    import time

    cfg = prog.config
    tiles = prog.tiles
    n = len(tiles)
    tm = Timings.empty(n, "simpy")
    env = simpy.Environment()
    mem = simpy.Resource(env, capacity=cfg.dram_channels)
    noc_r = simpy.Resource(env, capacity=1)
    noc_w = simpy.Resource(env, capacity=1)
    units = {u: simpy.Resource(env, capacity=1) for u in ("array", "vector", "transit")}
    buf = simpy.Container(env, capacity=cfg.buffer_bytes)
    ready = simpy.Store(env)
    done = simpy.Store(env)
    op_done = [env.event() for _ in prog.ops]
    first = {o.first_tile: o for o in prog.ops}
    for t in tiles:
        if t.alloc > cfg.buffer_bytes:
            raise ValueError(f"tile of op {t.op} needs {t.alloc} bytes, buffer is {cfg.buffer_bytes}")

    def transfer(dur, noc):
        with mem.request() as ch, noc.request() as link:
            yield ch
            yield link
            yield env.timeout(dur)

    def load_dma():
        for i, t in enumerate(tiles):
            tm.issue[i] = env.now
            o = first.get(i)
            if o is not None and o.deps:
                yield env.all_of([op_done[d] for d in o.deps])
            tm.dep[i] = env.now
            if t.alloc:
                yield buf.put(t.alloc)
            tm.alloc[i] = env.now
            if t.load_dur > 0:
                yield from transfer(t.load_dur, noc_r)
            tm.load_end[i] = env.now
            yield ready.put(i)

    def compute():
        for _ in range(n):
            i = yield ready.get()
            t = tiles[i]
            with units[t.unit].request() as u:
                yield u
                tm.comp_start[i] = env.now
                if t.comp_dur > 0:
                    yield env.timeout(t.comp_dur)
                tm.comp_end[i] = env.now
            free = t.alloc_in + (0 if t.stores else t.alloc_out)
            if free:
                yield buf.get(free)
            if t.stores:
                yield done.put(i)
            else:
                tm.store_start[i] = tm.store_end[i] = env.now

    def store_dma():
        while True:
            i = yield done.get()
            t = tiles[i]
            tm.store_start[i] = env.now
            if t.store_dur > 0:
                yield from transfer(t.store_dur, noc_w)
            tm.store_end[i] = env.now
            if t.alloc_out:
                yield buf.get(t.alloc_out)
            if t.last:
                op_done[t.op].succeed()

    env.process(load_dma())
    c = env.process(compute())
    env.process(store_dma())
    stores_left = [o for o in op_done]
    t0 = time.perf_counter()
    env.run(until=env.all_of([c, *stores_left]) if stores_left else c)
    wall = time.perf_counter() - t0
    return tm, SimStats(_event_count(env), wall)


def _event_count(env: simpy.Environment) -> int:
    # SimPy numbers every scheduled event with a running id (env._eid); it is the event count.
    eid = getattr(env, "_eid", None)
    try:
        return next(eid) if eid is not None else -1
    except TypeError:
        return -1
