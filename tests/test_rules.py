"""Cost rules on hand-built operators: every formula checked against arithmetic done by hand."""

from hypothesis import given
from hypothesis import strategies as st

from simfront.rules import apply_aten_rule, apply_onnx_rule
from simfront.trace import Op, TensorMeta


def T(shape, size=2, param=False, tid="x"):
    return TensorMeta(tid, tuple(shape), size, param)


def aten(name, ins, outs):
    return apply_aten_rule(Op(f"aten.{name}", ins, outs))


@given(st.integers(1, 300), st.integers(1, 300), st.integers(1, 300))
def test_mm_is_two_mkn(m, k, n):
    op = aten("mm", [T((m, k)), T((k, n), param=True)], [T((m, n))])
    assert op.category == "matmul"
    assert op.flops == 2 * m * k * n
    assert op.bytes == 2 * (m * k + k * n + m * n)
    assert op.weight_bytes == 2 * k * n


def test_bmm_addmm_linear_conv():
    assert aten("bmm", [T((4, 8, 16)), T((4, 16, 32))], [T((4, 8, 32))]).flops == 2 * 4 * 8 * 16 * 32
    assert aten("addmm", [T((32,)), T((8, 16)), T((16, 32))], [T((8, 32))]).flops == 2 * 8 * 16 * 32 + 8 * 32
    assert aten("linear", [T((2, 8, 16)), T((32, 16))], [T((2, 8, 32))]).flops == 2 * 2 * 8 * 16 * 32
    assert aten("linear", [T((8, 16)), T((32, 16)), T((32,))], [T((8, 32))]).flops == 2 * 8 * 16 * 32 + 8 * 32
    # conv2d: N=1, C_in=3, C_out=8, 3x3 kernel, 30x30 output
    conv = aten("convolution", [T((1, 3, 32, 32)), T((8, 3, 3, 3)), T((8,))], [T((1, 8, 30, 30))])
    assert conv.flops == 2 * (8 * 30 * 30) * (3 * 3 * 3) + 8 * 30 * 30


def test_fused_attention_counts_qk_av_and_softmax():
    q = T((1, 32, 512, 128))
    op = aten("_scaled_dot_product_flash_attention_for_cpu", [q, q, q], [T((1, 32, 512, 128)), T((1, 32, 512))])
    bhl = 32 * 512 * 512
    assert op.category == "attention"
    assert op.flops == 2 * bhl * 128 + 2 * bhl * 128 + 5 * bhl
    assert op.bytes_written == q.nbytes            # the log-sum-exp side output is not counted as traffic


def test_views_are_free_and_copies_are_not():
    x = T((64, 64))
    for name in ("view", "t", "transpose", "expand", "unsqueeze", "slice", "_unsafe_view"):
        op = aten(name, [x], [x])
        assert (op.category, op.flops, op.bytes) == ("view", 0, 0)
    c = aten("clone", [x], [x])
    assert (c.category, c.flops, c.bytes) == ("copy", 0, 2 * x.nbytes)


def test_elementwise_reduction_softmax():
    x = T((10, 20), size=4)
    assert aten("add", [x, x], [x]).flops == 200
    assert aten("add_", [x, x], [x]).category == "elementwise"     # in-place variant
    assert aten("mean", [x], [T((10, 1), size=4)]).flops == 200
    assert aten("_safe_softmax", [x], [x]).flops == 1000


def test_embedding_reads_rows_not_the_table():
    table, idx, out = T((128256, 4096), param=True), T((1, 7), size=8), T((1, 7, 4096))
    op = aten("embedding", [table, idx], [out])
    assert op.bytes_read == out.nbytes + idx.nbytes
    assert op.weight_bytes == out.nbytes
    assert op.flops == 0


def test_unknown_ops_are_zero_and_visible():
    op = aten("frobnicate", [T((4,))], [T((4,))])
    assert (op.category, op.flops, op.bytes) == ("unknown", 0, 0)


def test_onnx_rules():
    a, b, o = T((1, 2048, 4096)), T((4096, 1024), param=True), T((1, 2048, 1024))
    mm = apply_onnx_rule(Op("onnx.MatMul", [a, b], [o]))
    assert mm.flops == 2 * 2048 * 1024 * 4096 and mm.weight_bytes == b.nbytes
    gemm = apply_onnx_rule(Op("onnx.Gemm", [T((8, 16)), T((16, 32)), T((32,))], [T((8, 32))]))
    assert gemm.flops == 2 * 8 * 16 * 32 + 8 * 32
    # Transpose materialises in ONNX (a view in ATen); Reshape is free in both.
    assert apply_onnx_rule(Op("onnx.Transpose", [a], [a])).category == "copy"
    assert apply_onnx_rule(Op("onnx.Reshape", [a, T((3,), 8)], [a])).bytes == 0
    g = apply_onnx_rule(Op("onnx.Gather", [T((50257, 768), param=True), T((1, 9), 8)], [T((1, 9, 768))]))
    assert g.weight_bytes == 9 * 768 * 2
