"""Cost models: roofline arithmetic, the fusion bound, and the offload model's transfers."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from simfront.cost import Offload, Roofline, device, matmul_engine
from simfront.rules import apply_aten_rule
from simfront.trace import Op, TensorMeta, Trace


def T(tid, shape, param=False):
    return TensorMeta(tid, shape, 2, param)


def chain():
    """x -> mm(W1) -> a -> add(a, a) -> b -> mm(W2) -> c"""
    ops = [Op("aten.mm", [T("x", (64, 64)), T("w1", (64, 64), True)], [T("a", (64, 64))]),
           Op("aten.add", [T("a", (64, 64)), T("a", (64, 64))], [T("b", (64, 64))]),
           Op("aten.view", [T("b", (64, 64))], [T("bv", (64, 64))]),
           Op("aten.mm", [T("bv", (64, 64)), T("w2", (64, 64), True)], [T("c", (64, 64))])]
    return Trace("chain", "test", [apply_aten_rule(o) for o in ops])


def test_roofline_is_max_of_compute_and_memory_per_op():
    r = Roofline("r", flops_rate=1e12, byte_rate=1e11)
    rep = r.run(chain())
    mm = 2 * 64 ** 3 / 1e12, 3 * 64 * 64 * 2 / 1e11
    add = 64 * 64 / 1e12, 3 * 64 * 64 * 2 / 1e11
    assert rep.costs[0].time == pytest.approx(max(mm)) and rep.costs[0].bound == "compute"
    assert rep.costs[1].bound == "memory"
    assert rep.costs[1].time == pytest.approx(max(add))
    assert rep.costs[2].time == 0 and rep.costs[2].bound == "free"
    assert rep.time == pytest.approx(2 * max(mm) + max(add))


def test_from_device_uses_the_inference_simulators_rates():
    from disagg_sim.hardware import H100_SXM, LLAMA3_8B, CostModel

    r = device("h100")
    cm = CostModel(LLAMA3_8B, H100_SXM)
    assert (r.flops_rate, r.byte_rate) == (cm.flops_rate, cm.byte_rate)


def test_fused_bound_is_below_unfused(llama8b_prefill):
    unf, fus = device("h100").run(llama8b_prefill), device("h100", memory="fused").run(llama8b_prefill)
    assert fus.time < unf.time
    assert fus.trace.flops_of("matmul") == unf.trace.flops_of("matmul")
    # summing per-operator maxima is never faster than one roofline over the whole trace
    assert unf.time >= unf.notes["whole_trace_roofline_s"]


def test_offload_with_everything_supported_is_the_roofline(llama8b_prefill):
    acc = device("h100")
    every = frozenset({"matmul", "attention", "elementwise", "reduction", "softmax", "norm", "gather", "copy",
                       "creation"})
    off = Offload("all", acc, acc, link_bw=1e9, link_latency=1e-6, supported=every)
    a, b = acc.run(llama8b_prefill), off.run(llama8b_prefill)
    assert a.time == pytest.approx(b.time - sum(c.transfer for c in b.costs))
    assert sum(c.transfer for c in b.costs) == 0         # nothing ever needs to cross the link


def test_offload_moves_each_tensor_once_per_side():
    host = Roofline("h", 1e12, 1e11)
    off = Offload("o", Roofline("a", 1e13, 1e12), host, link_bw=1e10, link_latency=1e-6,
                  supported=frozenset({"matmul"}))
    rep = off.run(chain())
    nb = 64 * 64 * 2
    assert [c.unit for c in rep.costs] == ["accel", "host", "-", "accel"]
    assert rep.costs[0].transfer == 0                                     # x starts on the accelerator
    assert rep.costs[1].transfer == pytest.approx(1e-6 + nb / 1e10)       # a: accel -> host, read twice, moved once
    assert rep.costs[3].transfer == pytest.approx(1e-6 + nb / 1e10)       # b (through a view) -> accel


@given(st.floats(1e11, 1e15), st.floats(1e9, 1e13), st.floats(1.1, 10))
def test_faster_hardware_is_never_slower(f, b, k):
    tr = chain()
    slow, fast_c, fast_m = Roofline("s", f, b).run(tr), Roofline("c", f * k, b).run(tr), Roofline("m", f, b * k).run(tr)
    assert fast_c.time <= slow.time and fast_m.time <= slow.time


def test_matmul_engine_preset():
    e = matmul_engine("optical")
    assert e.supported == frozenset({"matmul"})
    assert e.link_bw == 64e9
