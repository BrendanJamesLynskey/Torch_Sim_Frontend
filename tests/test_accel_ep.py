"""Execution-provider partitioning, ONNX Runtime's own record of it, and the end-to-end routes."""

import pytest
import torch

from simfront import models
from simfront.accel import ep, lower, preset, report, simulate
from simfront.capture import export_onnx, trace_onnx


@pytest.fixture(scope="module")
def cnn_onnx(tmp_path_factory):
    path = tmp_path_factory.mktemp("cnn") / "cnn.onnx"
    m = models.tiny_cnn()
    ins, params = export_onnx(m, (models.image(),), path)
    return path, ins, params


@pytest.mark.req("SF-28")
def test_claim_assigns_every_node_once(cnn_onnx):
    import onnx

    path, *_ = cnn_onnx
    g = onnx.load(str(path)).graph
    part = ep.claim(path, {"Conv", "Relu", "MaxPool", "Gemm"})
    assert len(part.nodes) == len(g.node)
    assert part.claimed == sum(n.op_type in {"Conv", "Relu", "MaxPool", "Gemm"} for n in g.node)
    claimed = [n for sg in part.subgraphs for n in sg]
    assert len(claimed) == len(set(claimed)) == part.claimed
    # BatchNormalization stays on the host and splits the convolution chain into pieces
    assert len(part.subgraphs) > 1
    everything = ep.claim(path, {n.op_type for n in g.node})
    assert len(everything.subgraphs) == 1


@pytest.mark.req("SF-28")
def test_ort_profile_names_the_provider_of_every_node(cnn_onnx):
    import onnx

    path, ins, _ = cnn_onnx
    rows = ep.ort_providers(path, {ins[0]: models.image().numpy()})
    names = {n.name for n in onnx.load(str(path)).graph.node if n.op_type != "Constant"}
    assert names <= {r["node"] for r in rows}
    assert {r["provider"] for r in rows} == {"CPUExecutionProvider"}


def test_split_costs_add_up(cnn_onnx):
    path, ins, params = cnn_onnx
    tr = trace_onnx(path, ins, params, model_name="tiny-cnn")
    cfg = preset("edge-npu")
    alldev, rep = ep.simulate(tr, {"matmul", "elementwise", "norm", "reduction"}, cfg)
    assert alldev.crossings == 0 and alldev.host_s == 0
    assert alldev.device_s == pytest.approx(report(lower(tr, cfg), simulate(lower(tr, cfg))[0]).makespan)
    mm_only, _ = ep.simulate(tr, {"matmul"}, cfg)
    assert mm_only.crossings > 0 and mm_only.host_s > 0
    assert mm_only.total == pytest.approx(mm_only.device_s + mm_only.host_s + mm_only.link_s)


def test_export_and_onnx_routes_do_the_same_array_work(accel_traces, cnn_onnx):
    """End to end from both graph formats: the compute array performs the same multiply-accumulates."""
    path, ins, params = cnn_onnx
    on = lower(trace_onnx(path, ins, params), preset("edge-npu"))
    ex = lower(accel_traces["cnn"], preset("edge-npu"))
    assert sum(t.macs for t in on.tiles) == sum(t.macs for t in ex.tiles) > 0


def test_onnx_runtime_runs_the_cnn(cnn_onnx):
    import onnxruntime as ort

    path, ins, _ = cnn_onnx
    x = torch.randn(1, 3, 32, 32)
    want = models.tiny_cnn()(x).detach().numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {ins[0]: x.numpy()})[0]
    assert abs(got - want).max() < 1e-4


def test_cli_smoke(capsys, tmp_path):
    from simfront.accel.cli import main

    main(["--model", "tiny-cnn", "--timeline", str(tmp_path / "t.png"), "--chrome", str(tmp_path / "t.json")])
    out = capsys.readouterr().out
    assert "Hot-spot" in out and (tmp_path / "t.png").exists()
    main(["--workload", "polymul", "--log-n", "12", "--limbs", "2", "--transit", "1.6", "--engine", "fast"])
    assert "in-transit stage" in capsys.readouterr().out
    main(["--model", "tiny-cnn", "--engine", "cycle"])
    assert "cycles stepped" in capsys.readouterr().out
