"""Lowering: operators become tiles that fit the buffer and do exactly the operator's work."""

import math

import pytest

from simfront.accel.hw import AccelConfig, KiB, preset
from simfront.accel.lower import _split_gemm, gemm_dims, lower
from simfront.trace import Op, TensorMeta


def t(shape, param=False, itemsize=4, tid=None):
    return TensorMeta(tid or f"t{shape}", tuple(shape), itemsize, param)


@pytest.mark.req("SF-18")
def test_gemm_dims_match_the_flop_rules(accel_traces):
    """Every array operator's (batch, M, K, N) gives back the cost rule's matmul FLOPs (bias adds aside)."""
    seen = 0
    for tr in accel_traces.values():
        for o in tr.ops:
            if o.category != "matmul":
                continue
            dims = gemm_dims(o)
            assert dims, o.name
            macs = sum(b * m * k * n for b, m, k, n in dims)
            out = o.outputs[0].numel
            assert o.flops in (2 * macs, 2 * macs + out), o.name
            seen += 1
    assert seen >= 10


def test_conv_is_an_im2col_gemm():
    """Conv 3->16, 3x3, 32x32 output: M = 32*32 positions, K = 3*3*3, N = 16."""
    op = Op("aten.convolution", [t((1, 3, 32, 32)), t((16, 3, 3, 3), True), t((16,), True)], [t((1, 16, 32, 32))],
            category="matmul")
    assert gemm_dims(op) == [(1, 1024, 27, 16)]
    grouped = Op("aten.convolution", [t((1, 8, 8, 8)), t((8, 1, 3, 3), True)], [t((1, 8, 8, 8))], category="matmul")
    assert gemm_dims(grouped) == [(8, 64, 9, 1)]          # depthwise: 8 groups of one channel


@pytest.mark.req("SF-18")
@pytest.mark.parametrize("budget", [4 * KiB, 64 * KiB, 1 << 20])
def test_tiles_fit_and_cover_every_mac(accel_traces, budget):
    cfg = AccelConfig(buffer_bytes=2 * budget, tile_fraction=0.5)
    for tr in accel_traces.values():
        prog = lower(tr, cfg)
        assert not prog.unknown
        for o in prog.ops:
            tiles = prog.tiles[o.first_tile:o.first_tile + o.n_tiles]
            assert all(x.alloc <= budget for x in tiles)
            assert tiles[-1].last and tiles[-1].stores and not any(x.last for x in tiles[:-1])
            if o.unit == "array":
                op = tr.ops[o.index]
                assert sum(x.macs for x in tiles) == sum(b * m * k * n for b, m, k, n in gemm_dims(op))
                # each output element is stored exactly once, by the last K slice of its block
                assert sum(x.store_bytes for x in tiles) == op.outputs[0].nbytes
            else:
                op = tr.ops[o.index]
                assert sum(x.load_bytes for x in tiles) == op.bytes_read
                assert sum(x.store_bytes for x in tiles) == op.bytes_written


def test_splitting_k_refetches_operands():
    """A GEMM too big for one tile is split; tiling moves at least the operator's own bytes."""
    op = Op("aten.mm", [t((256, 512), tid="a"), t((512, 256), True, tid="w")], [t((256, 256), tid="c")],
            category="matmul", flops=2 * 256 * 512 * 256, bytes_read=4 * (256 * 512 * 2), bytes_written=4 * 256 * 256)
    from simfront.trace import Trace

    prog = lower(Trace("mm", "test", [op]), AccelConfig(buffer_bytes=128 * KiB))
    assert prog.n_tiles > 1
    assert prog.ops[0].tiled_bytes >= op.bytes


def test_split_gemm_halves_the_largest_dimension():
    bb, bm, bk, bn = _split_gemm(1, 1024, 64, 1024, 2, 2, 64 * KiB)
    assert 2 * (bm * bk + bk * bn) + 2 * bm * bn <= 64 * KiB
    assert bk == 64                                      # K was never the largest, so it was never split
    assert _split_gemm(16, 8, 8, 8, 2, 2, 1 << 20)[0] == 16   # small GEMMs batch together


@pytest.mark.req("SF-20")
def test_dependencies_follow_activations_not_weights(accel_traces):
    tr = accel_traces["cnn"]
    prog = lower(tr, preset("edge-npu"))
    assert prog.ops[0].deps == []                         # the first conv reads only the input image and weights
    for i, o in enumerate(prog.ops[1:], 1):
        assert o.deps and all(d < i for d in o.deps)


def test_unknown_operators_are_counted_not_simulated():
    from simfront.trace import Trace

    ops = [Op("aten.mystery", [t((4,))], [t((4,), tid="y")], category="unknown"),
           Op("aten.relu", [t((4,), tid="y")], [t((4,), tid="z")], category="elementwise", flops=4, bytes_read=16,
              bytes_written=16)]
    prog = lower(Trace("m", "test", ops), preset("edge-npu"))
    assert prog.unknown == {"aten.mystery": 1}
    assert len(prog.ops) == 1


def test_quantised_durations_are_whole_cycles(accel_traces):
    prog = lower(accel_traces["llama"], preset("edge-npu", quantise=True))
    for x in prog.tiles:
        for d in (x.load_dur, x.comp_dur, x.store_dur):
            assert d == math.floor(d) and d >= 0
