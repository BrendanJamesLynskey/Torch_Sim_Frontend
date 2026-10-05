"""A producer and a consumer joined by a bounded FIFO: back-pressure, measured.

The smallest model that shows the most important property of hardware queues. A
``simpy.Store(capacity=depth)`` is the FIFO: ``put`` blocks when it is full, so a producer
that runs ahead of its consumer is stalled (back-pressure) instead of filling an infinite
queue. The probe records the FIFO depth at every change, which gives the depth-over-time
plot, the time-averaged depth L, and with the mean time items spend in the FIFO (W) and
the rate they pass through it (lambda), a check of Little's law, L = lambda x W.

``examples/fifo_backpressure.py`` sweeps the depth and plots the traces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import simpy


@dataclass
class FifoRun:
    depth: int
    items: int
    end: float
    trace: list[tuple[float, int]]      # (time, items in the FIFO) after every change
    producer_blocked: float             # total time the producer waited on a full FIFO
    consumer_starved: float             # total time the consumer waited on an empty FIFO
    waits: list[float]                  # time each item spent in the FIFO

    @property
    def throughput(self) -> float:
        return self.items / self.end if self.end else 0.0

    @property
    def mean_depth(self) -> float:
        """Time-averaged FIFO occupancy, L."""
        area = 0.0
        for (t0, d), (t1, _) in zip(self.trace, self.trace[1:] + [(self.end, 0)], strict=True):
            area += d * (t1 - t0)
        return area / self.end if self.end else 0.0

    @property
    def mean_wait(self) -> float:
        return sum(self.waits) / len(self.waits) if self.waits else 0.0

    @property
    def max_depth(self) -> int:
        return max((d for _, d in self.trace), default=0)


def run(depth: int, items: int = 2000, produce: float = 1.0, consume: float = 1.0, burst: int = 1,
        jitter: float = 0.5, seed: int = 1) -> FifoRun:
    """A producer making ``items`` (``burst`` at a time every ``produce * burst`` time units on
    average) feeding a consumer that takes ``consume`` on average per item; service times are
    uniform in ``mean * (1 +/- jitter)`` from two independent seeded streams."""
    env = simpy.Environment()
    fifo = simpy.Store(env, capacity=depth)
    rng_p, rng_c = random.Random(seed), random.Random(seed + 1000)
    trace: list[tuple[float, int]] = [(0.0, 0)]
    stamps: dict[int, float] = {}
    waits: list[float] = []
    blocked = [0.0]
    starved = [0.0]

    def probe():
        if trace[-1][0] == env.now:
            trace[-1] = (env.now, len(fifo.items))
        else:
            trace.append((env.now, len(fifo.items)))

    def producer():
        for k in range(0, items, burst):
            yield env.timeout(produce * burst * rng_p.uniform(1 - jitter, 1 + jitter))
            for j in range(k, min(k + burst, items)):
                t0 = env.now
                yield fifo.put(j)
                blocked[0] += env.now - t0
                stamps[j] = env.now
                probe()

    def consumer():
        for _ in range(items):
            t0 = env.now
            j = yield fifo.get()
            starved[0] += env.now - t0
            waits.append(env.now - stamps.pop(j))
            probe()
            yield env.timeout(consume * rng_c.uniform(1 - jitter, 1 + jitter))

    env.process(producer())
    c = env.process(consumer())
    env.run(until=c)
    return FifoRun(depth, items, env.now, trace, blocked[0], starved[0], waits)
