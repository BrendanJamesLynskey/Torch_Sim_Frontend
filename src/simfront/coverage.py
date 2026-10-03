"""Operator coverage, as an engineering metric.

Two different questions, both worth tracking release over release:

* **Cost-model coverage:** which operators in a trace have a cost rule? An operator
  without one is costed at zero, which silently flatters the hardware, so the report
  lists every one by name.
* **Device coverage:** of the operators that cost something, which run on the
  accelerator and which fall back to the host? Counted three ways (operators, FLOPs,
  time), because they tell different stories: a matmul engine can cover 99.9% of FLOPs
  and a much smaller share of time.
"""

from __future__ import annotations

from collections import defaultdict

from .cost import CostReport
from .trace import Trace


def rule_coverage(trace: Trace) -> dict:
    types = trace.counts()
    unknown = trace.unknown()
    n = len(trace.ops)
    return {"op_types": len(types), "op_types_with_rule": len(types) - len(unknown), "ops": n,
            "ops_with_rule": n - sum(unknown.values()), "unknown": dict(unknown)}


def device_coverage(report: CostReport) -> dict:
    """Share of costed operators, FLOPs and time on the accelerator (transfers count against it)."""
    ops = [(o, c) for o, c in zip(report.trace.ops, report.costs, strict=True) if c.unit != "-"]
    on = [(o, c) for o, c in ops if c.unit == "accel"]
    flops = sum(o.flops for o, _ in ops) or 1
    total = report.time or 1.0
    return {"ops": len(on) / max(len(ops), 1), "flops": sum(o.flops for o, _ in on) / flops,
            "time": sum(c.time for _, c in on) / total,
            "host_time": sum(c.time for _, c in ops if c.unit == "host") / total,
            "transfer_time": sum(c.transfer for _, c in ops) / total}


def op_table(report: CostReport, top: int | None = None) -> list[dict]:
    """Per operator type: category, count, FLOPs, bytes, time, where it ran; sorted by time."""
    rows: dict[str, dict] = defaultdict(lambda: {"count": 0, "flops": 0, "bytes": 0, "time": 0.0, "units": set()})
    for o, c in zip(report.trace.ops, report.costs, strict=True):
        r = rows[o.name]
        r["category"] = o.category
        r["count"] += 1
        r["flops"] += o.flops
        r["bytes"] += o.bytes
        r["time"] += c.time + c.transfer
        r["units"].add(c.unit)
    out = sorted(({"op": k, **v, "units": "/".join(sorted(v["units"]))} for k, v in rows.items()),
                 key=lambda r: (-r["time"], -r["count"]))
    return out[:top] if top else out


def markdown(report: CostReport, top: int = 15) -> str:
    rc = rule_coverage(report.trace)
    lines = [f"Operator coverage: {report.trace.model}, {report.trace.route} trace, on {report.model}", "",
             f"* cost rules: {rc['op_types_with_rule']}/{rc['op_types']} operator types, "
             f"{rc['ops_with_rule']}/{rc['ops']} operator calls",
             f"* without a rule (costed at zero): {rc['unknown'] or 'none'}"]
    if any(c.unit == "host" for c in report.costs):
        dc = device_coverage(report)
        lines.append(f"* on the accelerator: {dc['ops']:.1%} of costed ops, {dc['flops']:.2%} of FLOPs, "
                     f"{dc['time']:.1%} of time (host {dc['host_time']:.1%}, transfers {dc['transfer_time']:.1%})")
    total = report.time or 1.0
    lines += ["", "| Operator | Category | Calls | GFLOP | MB moved | Time share | Unit |",
              "|---|---|---|---|---|---|---|"]
    for r in op_table(report, top):
        lines.append(f"| `{r['op']}` | {r['category']} | {r['count']} | {r['flops'] / 1e9:,.1f} | "
                     f"{r['bytes'] / 1e6:,.1f} | {r['time'] / total:.2%} | {r['units']} |")
    return "\n".join(lines)
