"""simfront.accel: run an operator trace through a SimPy model of an accelerator.

    from simfront.accel import preset, lower, simulate, report
    prog = lower(trace, preset("edge-npu"))        # operators -> tiles with durations
    timings, stats = simulate(prog)                  # SimPy: memory, NoC, DMA, buffer, compute
    print(report(prog, timings).to_markdown())       # latency, utilisation, stalls, hot-spot

The same program also runs on a compiled fast path (``fastpath.run``, C++ via pybind11,
bit-identical) and a cycle-stepped twin (``cycle.simulate_cycles``, identical in cycles).
See ``docs/accel_spec.md`` for the block's one-page specification.
"""

from .fastpath import run as run_fast
from .hw import PRESETS, AccelConfig, preset
from .lower import Program, Tile, lower
from .metrics import Report, report
from .sim import Timings, simulate

__all__ = ["PRESETS", "AccelConfig", "Program", "Report", "Tile", "Timings", "lower", "preset", "report",
           "run_fast", "simulate"]
