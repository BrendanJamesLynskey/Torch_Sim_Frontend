"""Cost rules: the FLOPs and bytes of each operator, for ATen and ONNX operators.

Every operator is put in a category, and the category decides how it is counted:

=============  ===============================================  ===========================
category       FLOPs                                            bytes
=============  ===============================================  ===========================
matmul         2 x (multiply-adds), plus bias adds              inputs + outputs
attention      QK^T and AV of a fused attention op (unmasked)   q, k, v + output
elementwise    one per output element                           inputs + outputs
reduction      one per input element (pooling included)         inputs + outputs
softmax        five per element (max, subtract, exp, sum, div)  inputs + outputs
norm           five per element                                 inputs + outputs
gather         none                                             indices + rows read + output
copy           none                                             inputs + outputs
creation       none                                             outputs (+ inputs if any)
view           none                                             none: metadata only
unknown        none, and reported by the coverage report        none
=============  ===============================================  ===========================

These are the counting conventions of a first-order roofline model, stated so they
can be argued with. Two choices matter most: attention FLOPs are counted *unmasked*
(a causal kernel that skips masked blocks does about half), and every non-view operator
is assumed to read its inputs from memory and write its outputs back (no fusion). The
cost model can apply an ideal-fusion bound on top (``simfront.cost``).

ATen views are free; the same reshapes and transposes in ONNX are separate nodes that
a runtime may or may not materialise. ONNX Transpose, Expand and Slice are counted as
copies, which is one reason byte totals differ between front ends while FLOPs agree.
"""

from __future__ import annotations

import math

from .trace import Op

# ── ATen ─────────────────────────────────────────────────────────────────
_ATEN = {
    "view": """view _unsafe_view reshape _reshape_alias t transpose permute expand expand_as unsqueeze squeeze
               slice select alias as_strided detach unflatten split split_with_sizes unbind chunk narrow view_as
               diagonal numpy_T lift_fresh lift_fresh_copy _assert_tensor_metadata _assert_async sym_size sym_numel
               sym_stride dropout feature_dropout""",
    "copy": """clone _to_copy to copy copy_ cat stack contiguous repeat repeat_interleave constant_pad_nd pad flip roll
               slice_scatter select_scatter index_put index_put_ index_copy index_copy_ scatter scatter_ gather
               masked_scatter _unsafe_index _local_scalar_dense""",
    "creation": """ones zeros full empty arange scalar_tensor full_like zeros_like ones_like empty_like empty_strided
                   new_ones new_zeros new_empty new_full fill fill_ zero_ rand randn randint randperm tril triu eye
                   linspace""",
    "elementwise": """add sub rsub mul div neg rsqrt sqrt pow exp exp2 log log1p tanh sigmoid silu gelu relu hardtanh
                      leaky_relu where eq ne lt le gt ge logical_not logical_and logical_or bitwise_and bitwise_or
                      bitwise_not bitwise_xor cos sin reciprocal abs clamp clamp_min clamp_max minimum maximum
                      masked_fill
                      erf isnan isinf sign floor ceil round square floor_divide remainder fmod lerp addcmul addcdiv
                      _safe_softmax_mask mish hardswish hardsigmoid elu sgn trunc""",
    "reduction": """sum mean amax amin max min prod var var_mean std any all argmax argmin cumsum cumprod norm
                    logsumexp linalg_vector_norm max_pool2d max_pool2d_with_indices max_pool1d max_pool3d avg_pool2d
                    avg_pool1d avg_pool3d adaptive_avg_pool2d _adaptive_avg_pool2d adaptive_max_pool2d""",
    "softmax": "_softmax softmax _safe_softmax _log_softmax log_softmax",
    "norm": """native_layer_norm layer_norm rms_norm _fused_rms_norm native_group_norm group_norm native_batch_norm
               batch_norm _native_batch_norm_legit_no_training _native_batch_norm_legit""",
    "matmul": "mm bmm addmm baddbmm matmul linear dot mv addmv _convolution convolution conv1d conv2d conv3d",
    "attention": """scaled_dot_product_attention _scaled_dot_product_flash_attention
                    _scaled_dot_product_flash_attention_for_cpu
                    _scaled_dot_product_efficient_attention _scaled_dot_product_cudnn_attention
                    _scaled_dot_product_fused_attention_overrideable""",
    "gather": "embedding index index_select",
}
ATEN_CATEGORY = {name: cat for cat, names in _ATEN.items() for name in names.split()}


def aten_packet(target: str) -> str:
    """``aten.mm.default`` -> ``mm``; ``aten.add_.Tensor`` -> ``add_``."""
    parts = target.split(".")
    return parts[1] if parts[0] == "aten" and len(parts) > 1 else target


def aten_category(packet: str) -> str:
    if packet.startswith("prim."):                                 # prim.device, prim.layout: metadata queries
        return "view"
    if packet in ATEN_CATEGORY:
        return ATEN_CATEGORY[packet]
    if packet.endswith("_") and packet[:-1] in ATEN_CATEGORY:      # in-place variants: add_, mul_
        return ATEN_CATEGORY[packet[:-1]]
    return "unknown"


def _numel(shape) -> int:
    return math.prod(shape)


def _matmul_flops(packet: str, op: Op) -> int:
    ins = [t.shape for t in op.inputs]
    out = op.outputs[0].shape if op.outputs else ()
    if packet in ("mm", "bmm", "matmul", "dot", "mv"):
        k = ins[0][-1] if ins[0] else 1
        return 2 * _numel(out) * k
    if packet in ("addmm", "baddbmm", "addmv"):                    # bias + a @ b
        k = ins[1][-1]
        return 2 * _numel(out) * k + _numel(out)
    if packet == "linear":                                         # x @ W^T (+ b)
        k = ins[0][-1]
        return 2 * _numel(out) * k + (_numel(out) if len(ins) > 2 else 0)
    if packet in ("_convolution", "convolution", "conv1d", "conv2d", "conv3d"):
        w = ins[1]                                                 # (C_out, C_in / groups, *kernel)
        flops = 2 * _numel(out) * _numel(w[1:])
        return flops + (_numel(out) if len(ins) > 2 and ins[2] else 0)
    raise KeyError(packet)


def _attention_flops(op: Op) -> int:
    q, k, v = (t.shape for t in op.inputs[:3])
    b_h_lq = _numel(q[:-1])                                        # batch x heads x query length
    lk = k[-2]
    return 2 * b_h_lq * lk * q[-1] + 2 * b_h_lq * lk * v[-1] + 5 * b_h_lq * lk


def apply_aten_rule(op: Op) -> Op:
    """Fill in category, FLOPs and bytes for an ATen operator (``op.name`` is ``aten.<packet>``)."""
    packet = op.name.split(".", 1)[1] if op.name.startswith("aten.") else op.name
    cat = aten_category(packet)
    op.category = cat
    rd = sum(t.nbytes for t in op.inputs)
    wr = sum(t.nbytes for t in op.outputs)
    op.weight_bytes = sum(t.nbytes for t in op.inputs if t.param)
    if cat in ("view", "unknown"):
        op.bytes_read = op.bytes_written = op.weight_bytes = 0
        return op
    op.bytes_read, op.bytes_written = rd, wr
    out_n = sum(t.numel for t in op.outputs)
    in_n = max((t.numel for t in op.inputs), default=0)
    if cat == "matmul":
        op.flops = _matmul_flops(aten_category_packet(packet), op)
    elif cat == "attention":
        op.flops = _attention_flops(op)
        op.bytes_read = sum(t.nbytes for t in op.inputs[:3])
        op.bytes_written = op.outputs[0].nbytes
    elif cat == "elementwise":
        op.flops = out_n
    elif cat == "reduction":
        op.flops = in_n
    elif cat in ("softmax", "norm"):
        op.flops = 5 * in_n
    elif cat == "gather":
        # embedding(weight, indices) / index(x, [idx]) / index_select(x, dim, idx): only the rows
        # that are looked up are read, never the whole table.
        rows = op.outputs[0].nbytes
        idx = sum(t.nbytes for t in op.inputs[1:])
        op.bytes_read = rows + idx
        op.weight_bytes = rows if op.inputs and op.inputs[0].param else 0
    elif cat == "creation":
        op.bytes_read = rd if packet in ("tril", "triu", "full_like", "zeros_like", "ones_like") else 0
    return op


def aten_category_packet(packet: str) -> str:
    return packet[:-1] if packet.endswith("_") and packet[:-1] in ATEN_CATEGORY else packet


# ── ONNX ─────────────────────────────────────────────────────────────────
_ONNX = {
    "view": "Reshape Unsqueeze Squeeze Identity Flatten Shape Size Constant Dropout SequenceAt",
    "copy": """Transpose Expand Slice Concat Cast CastLike Tile Pad ScatterND ScatterElements Split GatherND Trilu
               DepthToSpace SpaceToDepth SplitToSequence""",
    "creation": "ConstantOfShape Range EyeLike RandomNormal RandomUniform",
    "elementwise": """Add Sub Mul Div Neg Sqrt Reciprocal Pow Exp Log Tanh Sigmoid Relu Gelu Erf Where Equal Less
                      LessOrEqual Greater GreaterOrEqual And Or Not Xor IsNaN IsInf Sin Cos Abs Max Min Clip Sign Floor
                      Ceil Round Mod BitwiseAnd BitwiseOr BitwiseNot LeakyRelu HardSigmoid HardSwish Elu Mish""",
    "reduction": """ReduceMean ReduceSum ReduceMax ReduceMin ReduceProd ReduceL2 ArgMax ArgMin CumSum MaxPool
                    AveragePool GlobalAveragePool GlobalMaxPool""",
    "softmax": "Softmax LogSoftmax",
    "norm": "LayerNormalization SimplifiedLayerNormalization RMSNormalization BatchNormalization GroupNormalization",
    "matmul": "MatMul Gemm Conv",
    "attention": "Attention",
    "gather": "Gather",
}
ONNX_CATEGORY = {name: cat for cat, names in _ONNX.items() for name in names.split()}


def apply_onnx_rule(op: Op) -> Op:
    """Fill in category, FLOPs and bytes for an ONNX node (``op.name`` is ``onnx.<OpType>``)."""
    op_type = op.name.split(".", 1)[1]
    cat = ONNX_CATEGORY.get(op_type, "unknown")
    op.category = cat
    rd = sum(t.nbytes for t in op.inputs)
    wr = sum(t.nbytes for t in op.outputs)
    op.weight_bytes = sum(t.nbytes for t in op.inputs if t.param)
    if cat in ("view", "unknown"):
        op.bytes_read = op.bytes_written = op.weight_bytes = 0
        return op
    op.bytes_read, op.bytes_written = rd, wr
    ins = [t.shape for t in op.inputs]
    out = op.outputs[0].shape if op.outputs else ()
    out_n = sum(t.numel for t in op.outputs)
    in_n = max((t.numel for t in op.inputs), default=0)
    if cat == "matmul":
        if op_type == "MatMul":
            op.flops = 2 * _numel(out) * ins[0][-1]
        elif op_type == "Gemm":
            k = ins[0][0] if op.attrs.get("transA") else ins[0][-1]
            op.flops = 2 * _numel(out) * k + (_numel(out) if len(ins) > 2 else 0)
        else:                                                      # Conv: W is (C_out, C_in / group, *kernel)
            op.flops = 2 * _numel(out) * _numel(ins[1][1:]) + (_numel(out) if len(ins) > 2 else 0)
    elif cat == "attention":
        op.flops = _attention_flops(op)
    elif cat == "elementwise":
        op.flops = out_n
    elif cat == "reduction":
        op.flops = in_n
    elif cat in ("softmax", "norm"):
        op.flops = 5 * in_n
    elif cat == "gather":                                          # Gather(data, indices): rows only
        op.bytes_read = op.outputs[0].nbytes + sum(t.nbytes for t in op.inputs[1:])
        op.weight_bytes = op.outputs[0].nbytes if op.inputs[0].param else 0
    elif cat == "creation":
        op.bytes_read = 0
    return op
