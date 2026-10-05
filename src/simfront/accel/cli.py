"""simfront-accel: trace a model and run it through the SimPy accelerator model.

    simfront-accel --model tiny-cnn                                   # convolution net, torch.export route
    simfront-accel --model tiny-cnn --route onnx --timeline cnn.png   # the ONNX route, plus a timeline plot
    simfront-accel --model gpt2 --tokens 128 --preset dc-npu          # a real configuration, meta device
    simfront-accel --workload polymul --log-n 16 --limbs 24           # FHE-style NTT polynomial products
    simfront-accel --workload polymul --transit 1.6                   # ... with an in-transit NTT stage (speculative)
    simfront-accel --model tiny-cnn --engine cycle --quantise         # the cycle-stepped twin
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import warnings
from pathlib import Path

from .. import models
from .hw import PRESETS, MiB, preset
from .lower import lower
from .metrics import report

TINY = {"tiny-cnn", "tiny-llama", "tiny-gpt2"}


def capture(name: str, route: str, tokens: int):
    """A trace of ``name``: the tiny models are built on the CPU, the registry ones on the meta device."""
    import torch

    from ..capture import export_onnx, trace_dispatch, trace_export, trace_onnx
    from ..cli import capture as capture_registry

    if name not in TINY:
        return capture_registry(name, route, tokens, None)
    if name == "tiny-cnn":
        m, args, kw = models.tiny_cnn(), (models.image(),), {}
    else:
        cfg = models.tiny_llama() if name == "tiny-llama" else models.tiny_gpt2()
        m = models.build(cfg, device="cpu", dtype=torch.float32)
        args, kw = (models.tokens(1, tokens, "cpu"),), {"use_cache": False}
    if route == "dispatch":
        return trace_dispatch(m, *args, model_name=name, **kw)[0]
    if route == "export":
        return trace_export(m, args, kw, model_name=name)
    if route == "onnx":
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.onnx"
            ins, params = export_onnx(m, args, path, kw, weights=False)
            return trace_onnx(path, ins, params, model_name=name)
    raise SystemExit(f"route {route} is not supported here")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="simfront-accel", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tiny-cnn", choices=sorted(TINY | set(models.MODELS)))
    p.add_argument("--route", default="export", choices=["dispatch", "export", "onnx"])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--workload", default=None, choices=["polymul"], help="an FHE-style workload instead of a model")
    p.add_argument("--log-n", type=int, default=16)
    p.add_argument("--limbs", type=int, default=24)
    p.add_argument("--preset", default="edge-npu", choices=list(PRESETS))
    p.add_argument("--buffer-mib", type=float, default=None)
    p.add_argument("--dram-gbps", type=float, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--transit", type=float, default=None, metavar="OPS_PER_BYTE",
                   help="speculative in-transit stage in the read path for NTTs, with this compute budget")
    p.add_argument("--quantise", action="store_true", help="whole-cycle durations (needed by --engine cycle)")
    p.add_argument("--engine", default="simpy", choices=["simpy", "fast", "python", "cycle"])
    p.add_argument("--timeline", type=Path, default=None, help="write a timeline PNG")
    p.add_argument("--chrome", type=Path, default=None, help="write a Chrome trace (open in ui.perfetto.dev)")
    p.add_argument("--top", type=int, default=8)
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    kw = {}
    if a.buffer_mib:
        kw["buffer_bytes"] = int(a.buffer_mib * MiB)
    if a.dram_gbps:
        kw["dram_bw"] = a.dram_gbps * 1e9
    if a.channels:
        kw["dram_channels"] = a.channels
    if a.transit:
        kw["transit_ops_per_byte"] = a.transit
    if a.quantise or a.engine == "cycle":
        kw["quantise"] = True
    cfg = preset(a.preset, **kw)
    if a.workload == "polymul":
        from .ntt import polymul_trace

        trace = polymul_trace(a.log_n, a.limbs)
    else:
        trace = capture(a.model, a.route, a.tokens)
    prog = lower(trace, cfg)
    if a.engine == "simpy":
        from .sim import simulate

        tm, st = simulate(prog)
        cost = f"{st.events:,} events in {st.wall_s:.3f} s"
    elif a.engine == "cycle":
        from .cycle import simulate_cycles

        tm, cs = simulate_cycles(prog)
        cost = f"{cs.cycles:,} cycles stepped in {cs.wall_s:.3f} s"
    else:
        from . import fastpath

        tm = fastpath.run(prog, "python" if a.engine == "python" else "auto")
        cost = f"fast path ({tm.engine})"
    rep = report(prog, tm)
    print(f"{trace.model} via {trace.route}: {len(trace.ops)} operators -> {prog.n_tiles:,} tiles; {cost}")
    print()
    print(rep.to_markdown(a.top))
    if a.timeline:
        from .plot import timeline_png

        print(f"\ntimeline: {timeline_png(prog, tm, rep, a.timeline)}")
    if a.chrome:
        from .plot import chrome_trace

        print(f"chrome trace: {chrome_trace(prog, tm, a.chrome)}")


if __name__ == "__main__":
    main()
