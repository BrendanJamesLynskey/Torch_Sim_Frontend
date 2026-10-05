"""The fast path (Python recurrence and C++ module) is bit-identical to the SimPy model."""

import os
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simfront.accel import fastpath, lower, preset, simulate
from simfront.accel.hw import AccelConfig, KiB
from simfront.accel.lower import OpInfo, Program, Tile

REQUIRE_CPP = os.environ.get("SIMFRONT_REQUIRE_CPP") == "1"
cpp = pytest.mark.skipif(not fastpath.available() and not REQUIRE_CPP, reason="C++ fast path not built")

CONFIGS = [{}, {"buffer_bytes": 64 * KiB}, {"dram_channels": 4, "dram_bw": 100e9},
           {"array_rows": 4, "array_cols": 4, "dram_bw": 1e12, "noc_bw": 1e12, "buffer_bytes": 128 * KiB},
           {"quantise": True}]


def test_cpp_module_is_built_when_required():
    assert fastpath.available() or not REQUIRE_CPP


@pytest.mark.req("SF-22")
@pytest.mark.parametrize("name", ["cnn", "llama", "gpt2"])
@pytest.mark.parametrize("kw", CONFIGS)
def test_recurrence_equals_simpy_exactly(accel_traces, name, kw):
    prog = lower(accel_traces[name], preset("edge-npu", **kw))
    ref = simulate(prog)[0].as_tuple()
    assert fastpath.run_py(prog).as_tuple() == ref


@cpp
@pytest.mark.req("SF-22")
@pytest.mark.parametrize("name", ["cnn", "llama", "gpt2"])
@pytest.mark.parametrize("kw", CONFIGS)
def test_cpp_equals_simpy_exactly(accel_traces, name, kw):
    prog = lower(accel_traces[name], preset("edge-npu", **kw))
    assert fastpath.run_cpp(prog).as_tuple() == simulate(prog)[0].as_tuple()


def random_program(seed: int, n_ops: int, cap: int) -> Program:
    """Random tiles and dependencies, with awkward float durations and frequent ties."""
    rng = random.Random(seed)
    cfg = AccelConfig(buffer_bytes=cap)
    tiles, ops = [], []
    for o in range(n_ops):
        k = rng.randint(1, 4)
        deps = sorted(rng.sample(range(o), min(o, rng.randint(0, 2))))
        ops.append(OpInfo(o, "op", "matmul", "array", deps, len(tiles), k, 0, 0, 0))
        for j in range(k):
            last = j == k - 1
            stores = last or rng.random() < 0.5
            a_in = rng.choice([0, 1, cap // 4, cap // 3, cap // 2])
            a_out = rng.choice([0, 1, cap // 5, cap // 2 - a_in if a_in <= cap // 2 else 0])
            dur = [rng.choice([0.0, 1e-7, 1 / 3, 0.1, 0.2, 0.3, rng.random()]) for _ in range(3)]
            tiles.append(Tile(o, rng.choice(["array", "vector"]), 0, 0, a_in, a_out, dur[0], dur[1],
                              dur[2] if stores else 0.0, stores, last))
    return Program(cfg, "random", tiles, ops)


@pytest.mark.req("SF-22")
@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10**6), st.integers(1, 25), st.sampled_from([4, 100, 1000]))
def test_random_programs_agree(seed, n_ops, cap):
    prog = random_program(seed, n_ops, cap)
    ref = simulate(prog)[0].as_tuple()
    assert fastpath.run_py(prog).as_tuple() == ref
    if fastpath.available():
        assert fastpath.run_cpp(prog).as_tuple() == ref


@pytest.mark.req("SF-23")
def test_fast_path_refuses_a_shared_channel(accel_traces):
    prog = lower(accel_traces["cnn"], preset("edge-npu", dram_channels=1))
    with pytest.raises(ValueError, match="dram_channels"):
        fastpath.run_py(prog)
    if fastpath.available():
        with pytest.raises(ValueError, match="dram_channels"):
            fastpath.run_cpp(prog)
