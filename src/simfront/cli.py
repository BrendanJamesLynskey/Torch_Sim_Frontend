"""simfront: trace a model through one of the four front ends and cost it.

    simfront --model llama3-8b --tokens 2048                       # prefill, dispatch trace, H100 roofline
    simfront --model llama3-70b --decode 4096 --device h100 --n-devices 4
    simfront --model qwen2.5-0.5b --route onnx --offload optical   # matmul engine + host
    simfront --model gpt2 --route export --save trace.json
    simfront --model llama3-8b --fake-cpu                          # fused attention, as on a CPU
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import warnings
from pathlib import Path

from . import coverage, models
from .cost import device, matmul_engine


def capture(name: str, route: str, tokens: int, decode: int | None, batch: int = 1, fake_cpu: bool = False):
    """Build ``name`` on the meta device and trace one prefill (or one decode step) via ``route``.

    ``fake_cpu`` traces the dispatch route under fake tensors on the CPU instead (fused attention).
    """
    from .capture import export_onnx, trace_compile, trace_dispatch, trace_export, trace_onnx

    if fake_cpu:
        if route != "dispatch":
            raise SystemExit("--fake-cpu applies to the dispatch route")
        with models.fake_cpu():
            m = models.build(name, device="cpu")
            if decode is not None:
                kw = models.decode_inputs(m, decode, batch, device="cpu")
                wl = {"phase": "decode", "context": decode, "batch": batch, "device": "fake-cpu"}
                return trace_dispatch(m, model_name=name, workload=wl, **kw)[0]
            wl = {"phase": "prefill", "tokens": tokens, "batch": batch, "device": "fake-cpu"}
            return trace_dispatch(m, models.tokens(batch, tokens, "cpu"), use_cache=False, model_name=name,
                                  workload=wl)[0]
    m = models.build(name)
    if decode is not None:
        if route != "dispatch":
            raise SystemExit("decode steps carry a KV cache object; only the dispatch route traces them")
        kw = models.decode_inputs(m, decode, batch)
        return trace_dispatch(m, model_name=name, workload={"phase": "decode", "context": decode, "batch": batch},
                              **kw)[0]
    x = models.tokens(batch, tokens)
    wl = {"phase": "prefill", "tokens": tokens, "batch": batch}
    if route == "dispatch":
        return trace_dispatch(m, x, model_name=name, workload=wl)[0]
    if route == "export":
        return trace_export(m, (x,), {"use_cache": False}, model_name=name, workload=wl)
    if route == "compile":
        return trace_compile(m, (x,), {"use_cache": False}, model_name=name, workload=wl)[0]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.onnx"
        inputs, params = export_onnx(m, (x,), path, {"use_cache": False}, weights=False)
        return trace_onnx(path, inputs, params, model_name=name, workload=wl)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="simfront", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama3-8b", choices=models.MODELS)
    p.add_argument("--route", default="dispatch", choices=["dispatch", "export", "compile", "onnx"])
    p.add_argument("--tokens", type=int, default=2048, help="prefill length")
    p.add_argument("--decode", type=int, default=None, metavar="CONTEXT", help="trace one decode step instead")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="h100", choices=["h100", "a100", "optical"])
    p.add_argument("--n-devices", type=int, default=1)
    p.add_argument("--fake-cpu", action="store_true",
                   help="trace under fake CPU tensors (fused attention) instead of the meta device")
    p.add_argument("--fused", action="store_true", help="ideal-fusion memory bound")
    p.add_argument("--offload", default=None, choices=["h100", "a100", "optical"],
                   help="cost on a matmul-only device of this kind, with a host for everything else")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--save", type=Path, default=None, help="write the trace as JSON")
    a = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    trace = capture(a.model, a.route, a.tokens, a.decode, a.batch, a.fake_cpu)
    if a.save:
        a.save.write_text(trace.to_json())
    if a.offload:
        cm = matmul_engine(a.offload, n_devices=a.n_devices)
    else:
        cm = device(a.device, a.n_devices, memory="fused" if a.fused else "unfused")
    rep = cm.run(trace)
    print(f"{a.model} {trace.workload} via {trace.route}: {len(trace.ops)} ops captured in {trace.capture_s:.1f} s")
    print(f"  {trace.flops / 1e12:,.2f} TFLOP, {trace.bytes / 1e9:,.2f} GB moved, "
          f"weights read {trace.weight_bytes / 1e9:,.2f} GB")
    split = rep.bound_split()
    print(f"  {rep.model}: {rep.time * 1e3:,.3f} ms (compute-bound {split['compute']:.1%}, "
          f"memory-bound {split['memory']:.1%}, transfers {split['transfer']:.1%})")
    print()
    print(coverage.markdown(rep, a.top))


if __name__ == "__main__":
    main()
