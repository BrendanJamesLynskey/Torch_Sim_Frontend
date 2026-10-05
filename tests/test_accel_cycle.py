"""The cycle-stepped twin agrees with the event-driven model cycle for cycle."""

import pytest

from simfront.accel import fastpath, lower, preset, simulate
from simfront.accel.cycle import simulate_cycles
from simfront.accel.hw import KiB


@pytest.mark.req("SF-24")
@pytest.mark.parametrize("name", ["cnn", "llama"])
@pytest.mark.parametrize("kw", [{}, {"buffer_bytes": 64 * KiB}, {"array_rows": 4, "array_cols": 4, "dram_bw": 1e12}])
def test_cycle_twin_equals_event_driven(accel_traces, name, kw):
    prog = lower(accel_traces[name], preset("edge-npu", quantise=True, **kw))
    ev, stats = simulate(prog)
    cy, cs = simulate_cycles(prog)
    assert cy.as_tuple() == ev.as_tuple()
    assert cs.cycles == int(ev.makespan)
    assert cs.evaluations >= 3 * cs.cycles
    assert stats.events < cs.cycles                       # the event-driven model skips idle cycles


def test_cycle_twin_needs_quantised_durations(accel_traces):
    with pytest.raises(ValueError, match="quantise"):
        simulate_cycles(lower(accel_traces["cnn"], preset("edge-npu")))


def test_cycle_twin_matches_the_fast_path(accel_traces):
    prog = lower(accel_traces["cnn"], preset("edge-npu", quantise=True))
    assert simulate_cycles(prog)[0].as_tuple() == fastpath.run_py(prog).as_tuple()
