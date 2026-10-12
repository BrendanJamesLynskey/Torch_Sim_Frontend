"""SF-15: the CI gate fails when traced arithmetic changes, and only reports framework drift."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

CI = Path(__file__).resolve().parent.parent / "ci"
spec = importlib.util.spec_from_file_location("perf_gate", CI / "perf_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
BASE = json.loads((CI / "perf_baseline.json").read_text())


@pytest.mark.req("SF-15")
def test_gate_fails_on_an_arithmetic_change_and_tolerates_drift():
    assert gate.compare(BASE, BASE, 0.5)[1] == 0
    now = copy.deepcopy(BASE)
    k = next(iter(now["traces"]))
    now["traces"][k]["ops"] += 7                      # a library upgrade decomposes differently
    now["traces"][k]["bytes"] += 4096
    assert gate.compare(BASE, now, 0.5)[1] == 0       # reported as drift, not a failure ...
    assert gate.compare(BASE, now, 0.5, strict=True)[1] == 2     # ... unless --strict
    now["traces"][k]["flops"] += 1                    # one FLOP different: the arithmetic changed
    lines, bad = gate.compare(BASE, now, 0.5)
    assert bad == 1 and any("CHANGED" in x for x in lines)


@pytest.mark.req("SF-15")
def test_gate_allows_exactly_the_rotary_change_of_transformers_5_19():
    """The baseline was traced with the rotary frequencies as a matmul (2 FLOPs per product); from 5.19 they
    are an elementwise multiply (1 per product). The gate expects exactly that shift, and no other change."""
    assert BASE["rotary_matmul"] is True
    now = copy.deepcopy(BASE)
    now["rotary_matmul"] = False
    for k in now["traces"]:
        now["traces"][k]["flops"] -= gate.rotary_flops(k, True) - gate.rotary_flops(k, False)
    assert gate.compare(BASE, now, 0.5)[1] == 0
    assert gate.rotary_flops("llama3-8b prefill 512", False) == 64 * 512    # head_dim 128: 64 products/position
    assert gate.rotary_flops("llama3-8b decode @512", True) == 2 * 64       # one position
    assert gate.rotary_flops("gpt2 prefill 256", True) == 0                 # learned positions
    now["traces"]["llama3-8b prefill 512"]["flops"] += 1
    assert gate.compare(BASE, now, 0.5)[1] == 1


@pytest.mark.req("SF-13")
def test_gate_fails_when_capture_slows_down():
    now = copy.deepcopy(BASE)
    now["capture_s"] = BASE["capture_s"] * 1.6
    assert gate.compare(BASE, now, 0.5)[1] == 1
