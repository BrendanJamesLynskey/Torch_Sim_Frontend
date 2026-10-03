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


@pytest.mark.req("SF-13")
def test_gate_fails_when_capture_slows_down():
    now = copy.deepcopy(BASE)
    now["capture_s"] = BASE["capture_s"] * 1.6
    assert gate.compare(BASE, now, 0.5)[1] == 1
