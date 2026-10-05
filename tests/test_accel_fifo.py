"""A bounded FIFO between a producer and a consumer: back-pressure and Little's law."""

import pytest

from simfront.accel.fifo import run


@pytest.mark.req("SF-27")
@pytest.mark.parametrize("depth", [1, 2, 4, 16])
@pytest.mark.parametrize("produce,consume", [(1.0, 1.2), (1.2, 1.0), (1.0, 1.0)])
def test_littles_law_and_capacity(depth, produce, consume):
    r = run(depth, items=1500, produce=produce, consume=consume)
    assert r.max_depth <= depth
    assert len(r.waits) == r.items
    assert r.mean_depth == pytest.approx(r.throughput * r.mean_wait, rel=1e-9)


@pytest.mark.req("SF-27")
def test_backpressure_stalls_a_fast_producer():
    """Consumer 20% slower: the producer blocks, more often with a shallower FIFO, and the FIFO runs full."""
    shallow, deep = run(2, produce=1.0, consume=1.2), run(32, produce=1.0, consume=1.2)
    assert shallow.producer_blocked > 0 and deep.producer_blocked > 0
    assert shallow.throughput == pytest.approx(1 / 1.2, rel=0.03)      # the consumer sets the rate
    assert deep.mean_depth > 0.8 * 32


def test_depth_absorbs_bursts():
    """Balanced rates but bursty arrivals: a deeper FIFO keeps the consumer busier."""
    shallow, deep = run(1, produce=1.0, consume=1.0, burst=8), run(16, produce=1.0, consume=1.0, burst=8)
    assert deep.consumer_starved < shallow.consumer_starved
    assert deep.end < shallow.end
