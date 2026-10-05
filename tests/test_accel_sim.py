"""The SimPy model: back-pressure, dependencies, contention, metrics and the hot-spot."""

import pytest

from simfront.accel import lower, preset, report, simulate
from simfront.accel.hw import KiB, MiB
from simfront.accel.metrics import STALLS, histogram


def run(trace, **kw):
    prog = lower(trace, preset("edge-npu", **kw))
    tm, stats = simulate(prog)
    return prog, tm, stats


@pytest.mark.req("SF-19")
@pytest.mark.parametrize("buf", [64 * KiB, 256 * KiB, 2 * MiB])
def test_buffer_never_overflows(accel_traces, buf):
    prog, tm, _ = run(accel_traces["llama"], buffer_bytes=buf)
    rep = report(prog, tm)
    assert max(v for _, v in rep.buffer_steps) <= buf
    assert min(v for _, v in rep.buffer_steps) >= 0
    assert rep.buffer_steps[-1][1] == 0                   # everything freed at the end


@pytest.mark.req("SF-19")
def test_backpressure_stalls_loads_when_compute_is_slow(accel_traces):
    """A slow array and a fast memory: loads run ahead until the buffer is full, then wait."""
    prog, tm, _ = run(accel_traces["cnn"], array_rows=2, array_cols=2, dram_bw=1e12, noc_bw=1e12,
                      buffer_bytes=256 * KiB, tile_fraction=0.125)
    rep = report(prog, tm)
    waited = sum(tm.alloc[i] - tm.dep[i] for i in range(prog.n_tiles))
    assert waited > 0
    assert rep.hotspot == "compute array"


@pytest.mark.req("SF-20")
def test_loads_wait_for_producers_to_be_stored(accel_traces):
    prog, tm, _ = run(accel_traces["gpt2"])
    for o in prog.ops:
        for d in o.deps:
            dep_last = prog.ops[d].first_tile + prog.ops[d].n_tiles - 1
            assert tm.alloc[o.first_tile] >= tm.store_end[dep_last]


def test_pipeline_order_and_lower_bounds(accel_traces):
    prog, tm, _ = run(accel_traces["llama"], buffer_bytes=128 * KiB)
    n = prog.n_tiles
    for i in range(n):
        assert tm.issue[i] <= tm.dep[i] <= tm.alloc[i] <= tm.load_end[i] <= tm.comp_start[i] <= tm.comp_end[i]
        assert tm.comp_end[i] <= tm.store_start[i] <= tm.store_end[i]
        if i:
            assert tm.comp_start[i] >= tm.comp_end[i - 1]   # in-order, single issue
    assert tm.makespan >= sum(x.load_dur for x in prog.tiles)          # one load DMA
    assert tm.makespan >= sum(x.comp_dur for x in prog.tiles)


@pytest.mark.req("SF-21")
@pytest.mark.parametrize("name", ["cnn", "llama", "gpt2"])
@pytest.mark.parametrize("kw", [{}, {"buffer_bytes": 64 * KiB}, {"dram_bw": 1e12, "array_rows": 4, "array_cols": 4}])
def test_stalls_sum_to_the_makespan(accel_traces, name, kw):
    prog, tm, _ = run(accel_traces[name], **kw)
    rep = report(prog, tm)
    assert set(rep.stalls) == {"compute", *STALLS}
    assert sum(rep.stalls.values()) == pytest.approx(rep.makespan, rel=1e-12)
    for c in rep.components.values():
        assert 0.0 <= c["busy"] <= 1.0 + 1e-12


@pytest.mark.req("SF-21")
def test_hotspot_moves_with_the_bottleneck(accel_traces):
    tr = accel_traces["llama"]
    slow_mem = report(*run(tr, dram_bw=2e9)[:2])
    slow_noc = report(*run(tr, dram_bw=1e12, noc_bw=2e9)[:2])
    slow_pe = report(*run(tr, dram_bw=1e12, noc_bw=1e12, array_rows=2, array_cols=2)[:2])
    assert slow_mem.hotspot == "off-chip memory"
    assert slow_noc.hotspot == "NoC read"
    assert slow_pe.hotspot == "compute array"


def test_one_shared_channel_makes_loads_and_stores_contend(accel_traces):
    """Same per-transfer bandwidth; one channel means a store can delay the next load."""
    tr = accel_traces["llama"]
    two = simulate(lower(tr, preset("edge-npu", dram_channels=2, dram_bw=25.6e9)))[0]
    one = simulate(lower(tr, preset("edge-npu", dram_channels=1, dram_bw=12.8e9)))[0]
    assert one.makespan > two.makespan


def test_deterministic(accel_traces):
    a = simulate(lower(accel_traces["cnn"], preset("edge-npu")))[0]
    b = simulate(lower(accel_traces["cnn"], preset("edge-npu")))[0]
    assert a.as_tuple() == b.as_tuple()


@pytest.mark.req("SF-21")
def test_outputs_report_histogram_timeline_and_chrome_trace(accel_traces, tmp_path):
    import json

    from simfront.accel.plot import chrome_trace, timeline_png

    prog, tm, _ = run(accel_traces["cnn"])
    rep = report(prog, tm)
    md = rep.to_markdown()
    for text in ("off-chip memory", "compute array", "vector unit", "on-chip buffer", "Hot-spot", "load bandwidth"):
        assert text in md
    hist = rep.op_latency_histogram(6)
    assert sum(c for _, _, c in hist) == len(rep.ops)
    png = timeline_png(prog, tm, rep, tmp_path / "t.png")
    assert png.stat().st_size > 10_000 and png.read_bytes()[:4] == b"\x89PNG"
    ev = json.loads(chrome_trace(prog, tm, tmp_path / "t.json").read_text())["traceEvents"]
    assert sum(e["ph"] == "X" for e in ev) >= prog.n_tiles


def test_histogram_bins():
    h = histogram([1, 10, 100, 1000], bins=3)
    assert [c for *_, c in h] == [1, 1, 2]
    assert histogram([]) == []
