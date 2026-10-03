"""Front end 4: walk an ONNX graph with inferred shapes.

ONNX is the framework-neutral route: models exported from PyTorch, TensorFlow or JAX
arrive as a graph of standard operators with versioned semantics. Shape inference with
``data_prop=True`` propagates shape *values* through ``Shape``/``Concat``/``Reshape``
chains, which an exported transformer needs: without it the attention MatMuls of an
exported Llama have unknown dimensions.

``export_onnx`` exports a PyTorch model without its weights: the model may live on the
meta device, the weights stay out of the file (they become typed graph inputs instead),
and so an 8B-parameter model exports to about 11 MB.
"""

from __future__ import annotations

import time
from pathlib import Path

import onnx
from onnx import helper, numpy_helper, shape_inference

from ..rules import apply_onnx_rule
from ..trace import Op, TensorMeta, Trace


def export_onnx(model, args: tuple, path: str | Path, kwargs: dict | None = None, *,
                weights: bool = True) -> tuple[list[str], set[str]]:
    """Export with the torch.export-based exporter; returns (the model's real inputs, its parameter names).

    ``weights=False`` drops the initializers and keeps them as typed graph inputs, so their
    shapes survive for the cost model.

    The exporter's ONNX optimizer is skipped in both cases: its constant folding needs real
    weight values (a meta-device model has none), and it renames the weights it pre-transposes.
    ONNX Runtime optimises the graph itself when a session is created, and :func:`trace_onnx`
    folds constant subgraphs as a runtime would.

    The parameter names matter because ONNX does not distinguish parameters from buffers (a
    causal-mask table, rotary frequencies): both are initializers, named after the module path.
    ``remove_duplicate=False`` keeps every alias of a tied weight (GPT-2's ``lm_head.weight`` is
    ``transformer.wte.weight``).
    """
    import torch

    prog = torch.onnx.export(model, args, kwargs=kwargs or {}, dynamo=True, optimize=False, verbose=False)
    if weights:
        prog.save(str(path))
    else:
        prog.save(str(path), include_initializers=False, keep_initializers_as_inputs=True)
    inits = set(prog.model.graph.initializers)
    inputs = [v.name for v in prog.model.graph.inputs if v.name not in inits]
    return inputs, {n for n, _ in model.named_parameters(remove_duplicate=False)}


# Folding these over a weight gives the same weight rearranged; anything else gives derived data.
LAYOUT = {"Transpose", "Reshape", "Cast", "CastLike", "Identity", "Unsqueeze", "Squeeze", "Flatten", "Expand"}


def _itemsize(elem_type: int) -> int:
    return helper.tensor_dtype_to_np_dtype(elem_type).itemsize


def _shape(tt) -> tuple[int, ...] | None:
    if not tt.HasField("shape"):
        return None
    dims = [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim]
    return None if any(d is None for d in dims) else tuple(dims)


def trace_onnx(model: str | Path | onnx.ModelProto, user_inputs: list[str] | None = None,
               params: set[str] | None = None, *, model_name: str = "", workload: dict | None = None) -> Trace:
    """Operators of an ONNX model's main graph, with shapes from ``infer_shapes(data_prop=True)``.

    Constant data is the initializers, plus (when ``user_inputs`` is given) every graph input that
    is not a real model input: that is how a weightless export carries them. Weights are the
    constants named in ``params`` (from :func:`export_onnx`), or every constant if it is None.
    """
    t0 = time.perf_counter()
    m = onnx.load(str(model)) if not isinstance(model, onnx.ModelProto) else model
    if m.functions:
        from onnx import inliner
        m = inliner.inline_local_functions(m)
    m = shape_inference.infer_shapes(m, data_prop=True)
    g = m.graph
    inits = {t.name for t in g.initializer}
    consts = set(inits) | ({v.name for v in g.input} - set(user_inputs) if user_inputs is not None else set())
    params = set(consts) if params is None else consts & set(params)
    meta: dict[str, tuple[tuple[int, ...] | None, int]] = {}
    for v in [*g.input, *g.value_info, *g.output]:
        tt = v.type.tensor_type
        meta[v.name] = (_shape(tt), _itemsize(tt.elem_type) if tt.elem_type else 0)
    for t in g.initializer:
        meta[t.name] = (tuple(t.dims), _itemsize(t.data_type))
    for n in g.node:                                       # Constant outputs are not in value_info
        if n.op_type == "Constant" and n.attribute and n.attribute[0].type == onnx.AttributeProto.TENSOR:
            arr = numpy_helper.to_array(n.attribute[0].t)
            meta[n.output[0]] = (tuple(arr.shape), arr.dtype.itemsize)
        elif n.op_type == "SplitToSequence":               # a sequence holds the same bytes as its input
            meta[n.output[0]] = meta.get(n.input[0], (None, 0))

    # Nodes whose inputs are all constants (a Transpose of a weight, a Cast of a buffer) are folded
    # by any runtime when the model is loaded; they cost nothing per inference, and their outputs
    # are still weights. A weightless export leaves them in the graph; an optimised one folds them.
    const = consts | {n.output[0] for n in g.node if n.op_type == "Constant"}
    ops = []
    for n in g.node:
        unresolved = False
        folded = n.op_type != "Constant" and bool(n.input) and all(x in const for x in n.input if x)
        if folded:
            const.update(x for x in n.output if x)
            if n.op_type in LAYOUT and any(x in params for x in n.input):    # the same weight, rearranged
                params.update(x for x in n.output if x)

        def tm(name):
            nonlocal unresolved
            shape, size = meta.get(name, (None, 0))
            if shape is None:
                unresolved = True
                shape = ()
            return TensorMeta(name, shape, size, name in params)

        ins = [tm(x) for x in n.input if x]
        outs = [tm(x) for x in n.output if x]
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute
                 if a.type in (onnx.AttributeProto.INT, onnx.AttributeProto.FLOAT, onnx.AttributeProto.STRING)}
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in attrs.items()}
        op = Op(f"onnx.{n.op_type}", ins, outs, attrs)
        if folded:
            op.category, op.attrs["folded"] = "view", True
        elif unresolved and n.op_type != "Constant":
            op.attrs["unresolved_shape"] = True                # reported by the coverage report, costed at zero
            op.category = "unknown"
        else:
            apply_onnx_rule(op)
        ops.append(op)
    return Trace(model_name or g.name or "onnx", "onnx", ops, workload or {}, time.perf_counter() - t0)
