import pytest
from onnx import TensorProto, helper

from simfront import coverage
from simfront.capture import trace_onnx
from simfront.cost import device, matmul_engine


@pytest.mark.req("SF-06")
def test_unknown_operator_is_reported(tmp_path):
    """A custom-domain op has no rule: it must be named in the report, not silently costed at zero."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
    w = helper.make_tensor("w", TensorProto.FLOAT, [8, 8], [0.0] * 64)
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["h"]),
                           helper.make_node("FancyGelu", ["h"], ["y"], domain="vendor")], "g", [x], [y], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("vendor", 1)])
    tr = trace_onnx(m)
    rc = coverage.rule_coverage(tr)
    assert rc["unknown"] == {"onnx.FancyGelu": 1}
    assert (rc["op_types"], rc["op_types_with_rule"]) == (2, 1)
    assert tr.flops == 2 * 4 * 8 * 8 and tr.weight_bytes == 8 * 8 * 4
    assert "onnx.FancyGelu" in coverage.markdown(device("h100").run(tr))


@pytest.mark.req("SF-12")
def test_device_coverage_three_ways(llama8b_prefill):
    rep = matmul_engine("optical").run(llama8b_prefill)
    dc = coverage.device_coverage(rep)
    assert dc["flops"] > 0.99                       # nearly every FLOP is a matmul ...
    assert dc["time"] < 0.5                         # ... and still most of the time is elsewhere
    assert abs(dc["time"] + dc["host_time"] + dc["transfer_time"] - 1) < 1e-9
    assert sum(r["time"] for r in coverage.op_table(rep)) == pytest.approx(rep.time)


@pytest.mark.req("SF-06")
def test_rule_coverage_of_real_traces_is_complete(llama8b_prefill):
    assert coverage.rule_coverage(llama8b_prefill)["unknown"] == {}
