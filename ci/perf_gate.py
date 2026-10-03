"""Behaviour and speed gate against ci/perf_baseline.json.

* Arithmetic: the FLOPs and weight bytes of each reference trace are properties of the model,
  not of the framework, so they must not change at all. A change fails the gate.
* Framework drift: operator counts and bytes moved depend on how PyTorch and transformers
  decompose the model. A library upgrade may change them; the gate reports every change
  (this is how an upgrade's effect on the trace is noticed) and fails only with --strict.
* Speed: capturing Llama-3-8B's prefill trace must not take more than --margin (default 50%)
  longer than the baseline (best of three).

    python ci/perf_gate.py [--margin 0.5] [--strict] [--bless]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

from simfront import models
from simfront.capture import trace_dispatch

BASE = Path(__file__).parent / "perf_baseline.json"
CASES = {"llama3-8b prefill 512": ("llama3-8b", 512, None), "llama3-8b decode @512": ("llama3-8b", None, 512),
         "qwen2.5-0.5b prefill 256": ("qwen2.5-0.5b", 256, None), "gpt2 prefill 256": ("gpt2", 256, None)}


def trace(name, tokens, ctx):
    m = models.build(name)
    if ctx is not None:
        return trace_dispatch(m, **models.decode_inputs(m, ctx))[0]
    return trace_dispatch(m, models.tokens(1, tokens))[0]


def measure() -> dict:
    out = {"traces": {}}
    for k, case in CASES.items():
        t = trace(*case)
        out["traces"][k] = {"flops": t.flops, "weight_bytes": t.weight_bytes, "ops": len(t.ops), "bytes": t.bytes}
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        trace("llama3-8b", 2048, None)
        best = min(best, time.perf_counter() - t0)
    out["capture_s"] = best
    return out


def compare(base: dict, now: dict, margin: float, strict: bool = False) -> tuple[list[str], int]:
    """The gate's verdict: (report lines, number of failures)."""
    lines = ["# simfront gate", "", "| check | baseline | now | verdict |", "|---|---|---|---|"]
    bad = 0
    for k, b in base["traces"].items():
        n = now["traces"][k]
        for f in ("flops", "weight_bytes"):
            ok = n[f] == b[f]
            bad += not ok
            lines.append(f"| {k}: {f} | {b[f]:,} | {n[f]:,} | {'ok' if ok else 'CHANGED'} |")
        for f in ("ops", "bytes"):
            ok = n[f] == b[f]
            bad += strict and not ok
            lines.append(f"| {k}: {f} | {b[f]:,} | {n[f]:,} | {'ok' if ok else 'drift (framework)'} |")
    b, n = base["capture_s"], now["capture_s"]
    ok = n <= (1 + margin) * b
    bad += not ok
    lines.append(f"| capture llama3-8b prefill 2048 (s) | {b:.3f} | {n:.3f} | {'ok' if ok else 'SLOWER'} "
                 f"(margin {margin:.0%}) |")
    return lines, bad


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--strict", action="store_true", help="fail on framework drift too")
    ap.add_argument("--bless", action="store_true")
    a = ap.parse_args()
    now = measure()
    if a.bless:
        BASE.write_text(json.dumps(now, indent=1) + "\n")
        print("blessed", now)
        return 0
    lines, bad = compare(json.loads(BASE.read_text()), now, a.margin, a.strict)
    Path("perf_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
