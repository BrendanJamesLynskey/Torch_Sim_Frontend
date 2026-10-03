"""Nightly sweep: every model x phase x length, costed on three devices, unfused and fused, to sweep.csv."""

import csv
import logging
import warnings

from simfront import coverage, models
from simfront.capture import trace_dispatch
from simfront.cost import device, matmul_engine

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
with open("sweep.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "phase", "length", "ops", "tflop", "gb_moved", "device", "memory", "time_ms",
                "compute_bound_share", "matmul_engine_ms", "matmul_engine_time_on_accel"])
    for name in ("llama3-8b", "mistral-7b", "qwen2.5-0.5b", "gpt2"):
        m = models.build(name)
        for phase in ("prefill", "decode"):
            for n in (128, 512, 1024):
                if phase == "prefill":
                    t = trace_dispatch(m, models.tokens(1, n))[0]
                else:
                    t = trace_dispatch(m, **models.decode_inputs(m, n))[0]
                eng = matmul_engine("optical").run(t)
                for dev in ("h100", "a100", "optical"):
                    for mem in ("unfused", "fused"):
                        r = device(dev, memory=mem).run(t)
                        w.writerow([name, phase, n, len(t.ops), f"{t.flops / 1e12:.4f}", f"{t.bytes / 1e9:.3f}",
                                    dev, mem, f"{r.time * 1e3:.4f}", f"{r.bound_split()['compute']:.4f}",
                                    f"{eng.time * 1e3:.3f}", f"{coverage.device_coverage(eng)['time']:.4f}"])
print(open("sweep.csv").read())
