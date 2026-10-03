"""Every number quoted in the README and in deck SimEng 10.

    python examples/results.py                    # writes examples/results.md (about 3 minutes, 2 GB RAM)

Models are built from their published configurations on the meta device (no weights).
Device numbers come from Disaggregated_Inference_Sim (datasheet-level H100; the optical
part is hypothetical); the host CPU and its link are illustrative. See src/simfront/cost.py.
"""

from __future__ import annotations

import logging
import platform
import resource
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import onnx
import torch
import transformers
from disagg_sim.hardware import H100_SXM, LINKS, CostModel
from torch.utils.flop_counter import FlopCounterMode

from simfront import coverage, models
from simfront.capture import export_onnx, trace_compile, trace_dispatch, trace_export, trace_onnx
from simfront.cost import HOST_CPU, Offload, device

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
HERE = Path(__file__).parent
ROOT = HERE.parent
OUT: list[str] = []
T, C = 2048, 2048          # prefill length; decode context


def h(title):
    OUT.append(f"\n## {title}\n")


def p(text=""):
    OUT.append(text)


def table(head, rows):
    OUT.append("| " + " | ".join(head) + " |")
    OUT.append("|" + "---|" * len(head))
    for r in rows:
        OUT.append("| " + " | ".join(str(x) for x in r) + " |")
    OUT.append("")


def g(x, unit=1e9, nd=2):
    return f"{x / unit:,.{nd}f}"


def ms(x):
    return f"{x * 1e3:,.2f}"


def prefill_meta(name, tokens=T):
    m = models.build(name)
    return trace_dispatch(m, models.tokens(1, tokens), model_name=name,
                          workload={"phase": "prefill", "tokens": tokens})[0]


def prefill_fake(name, tokens=T):
    with models.fake_cpu():
        m = models.build(name, device="cpu")
        return trace_dispatch(m, models.tokens(1, tokens, "cpu"), use_cache=False, model_name=name,
                              workload={"phase": "prefill", "tokens": tokens})[0]


def decode_meta(name, ctx=C):
    m = models.build(name)
    return trace_dispatch(m, model_name=name, workload={"phase": "decode", "context": ctx},
                          **models.decode_inputs(m, ctx))[0]


def decode_fake(name, ctx=C):
    with models.fake_cpu():
        m = models.build(name, device="cpu")
        return trace_dispatch(m, model_name=name, workload={"phase": "decode", "context": ctx},
                              **models.decode_inputs(m, ctx, device="cpu"))[0]


def main() -> None:
    t_start = time.perf_counter()
    p("# Torch_Sim_Frontend: recorded results")
    p()
    p("Written by `examples/results.py`. Every number in the README and in deck SimEng 10 comes from this file.")
    p("Models are built from published configurations on the meta device or under fake tensors: no weights are")
    p("downloaded or allocated. Prefill is one 2,048-token prompt; decode is one step after a 2,048-token context;")
    p("batch 1, BF16. Device rates are Disaggregated_Inference_Sim's H100-SXM roofline (peak x efficiency).")

    h("1. Environment")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    table(["Item", "Value"], [["Python", platform.python_version()], ["PyTorch", torch.__version__],
                              ["transformers", transformers.__version__], ["onnx", onnx.__version__],
                              ["CPU", platform.processor() or platform.machine()], ["simfront commit", git or "n/a"]])

    # ── 2. real models, no weights ───────────────────────────────────────
    h("2. Real model configurations traced without weights")
    p("Dispatch trace on the meta device, one 2,048-token prefill. Parameters are counted from the instantiated")
    p("model; every parameter is a meta tensor (checked), so none of this used parameter memory.")
    p()
    rows = []
    for name in models.MODELS:
        m = models.build(name)
        n = sum(t.numel() for t in m.parameters())
        assert all(t.is_meta for t in m.parameters())
        tr = trace_dispatch(m, models.tokens(1, T), model_name=name)[0]
        rc = coverage.rule_coverage(tr)
        rows.append([name, g(n, 1e9, 2) + " B", f"{len(tr.ops):,}", rc["op_types"], f"{tr.capture_s:.2f}",
                     g(tr.flops, 1e12, 2), g(tr.weight_bytes, 1e9, 2), rc["op_types_with_rule"]])
        del m
    table(["Model", "Parameters", "Operators", "Op types", "Capture (s)", "TFLOP", "Weights read (GB)",
           "Types with a cost rule"], rows)
    p("Configurations: the architectural fields of the published Hugging Face `config.json` files (sources in")
    p("`src/simfront/configs/`). Llama-3 configs are from the NousResearch mirror, which is not gated.")

    # ── 3. four front ends ──────────────────────────────────────────────
    h("3. Four front ends, one model: Llama-3-8B prefill")
    p("The same model, on the meta device unless stated, captured four ways. FLOPs are the trace's total; matmul")
    p("FLOPs include the attention matmuls (QK^T, AV) where attention is decomposed.")
    p()
    m = models.build("llama3-8b")
    x = models.tokens(1, T)
    routes = {"dispatch (meta)": trace_dispatch(m, x)[0],
              "torch.export + Core ATen": trace_export(m, (x,), {"use_cache": False}),
              "torch.compile backend": None, "ONNX (exported without weights)": None}
    routes["torch.compile backend"], n_graphs = trace_compile(m, (x,), {"use_cache": False})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "llama3-8b.onnx"
        t0 = time.perf_counter()
        names, params = export_onnx(m, (x,), path, {"use_cache": False}, weights=False)
        export_s = time.perf_counter() - t0
        onnx_mb = path.stat().st_size / 1e6
        routes["ONNX (exported without weights)"] = trace_onnx(path, names, params)
    routes["dispatch (fake CPU tensors)"] = prefill_fake("llama3-8b")
    rows = []
    for k, tr in routes.items():
        rows.append([k, f"{len(tr.ops):,}", len(tr.counts()), f"{tr.flops_of('matmul'):,}",
                     f"{tr.flops_of('attention'):,}", g(tr.flops, 1e12, 3), g(tr.bytes), g(tr.weight_bytes, 1e9, 3),
                     f"{tr.capture_s:.1f}"])
    table(["Front end", "Operators", "Op types", "Matmul FLOPs", "Fused-attention FLOPs", "Total TFLOP",
           "Bytes moved (GB)", "Weights read (GB)", "Capture (s)"], rows)
    spec = models.analytic_spec("llama3-8b")
    an_w, an_att = 2 * spec.matmul_params * T, 4 * spec.n_layers * spec.d_model * T * T
    rope = 2 * (spec.head_dim // 2) * T
    onnx_tr = routes["ONNX (exported without weights)"]
    p(f"* The four meta-device routes agree exactly on matmul FLOPs, {routes['dispatch (meta)'].flops_of('matmul'):,}"
      f" (ONNX: {onnx_tr.flops_of('matmul'):,}, the rotary-frequency matmul of {rope:,} FLOPs constant-folded),")
    p(f"  and on weights read, {routes['dispatch (meta)'].weight_bytes:,} bytes.")
    p(f"* That is 2 x matmul parameters x tokens ({an_w:,}) + unmasked attention 4 x layers x d x T^2 ({an_att:,})"
      f" + the rotary matmul ({rope:,}).")
    p(f"* torch.compile captured {n_graphs} graph(s), no graph breaks. The ONNX export (without weights) took"
      f" {export_s:.1f} s and is {onnx_mb:.1f} MB.")
    p("* Under fake CPU tensors attention is one fused operator per layer"
      " (`_scaled_dot_product_flash_attention_for_cpu`); its FLOPs include the softmax (5 per score).")
    p()
    p("Bytes moved, by operator category (GB):")
    p()
    cats = ["matmul", "attention", "softmax", "elementwise", "copy", "reduction", "norm", "gather", "creation"]
    rows = []
    for k, tr in routes.items():
        bc = tr.by_category()
        rows.append([k] + [g(bc.get(c, {}).get("bytes", 0)) for c in cats])
    table(["Front end"] + cats, rows)
    p("ATen views are free. ONNX Transpose, Expand and Slice materialise and are counted as copies; the ATen")
    p("decompositions differ in where they upcast to FP32 and copy. The arithmetic agrees; the memory traffic is a")
    p("property of the decomposition, not of the model.")

    # ── 4. closed forms ─────────────────────────────────────────────────
    h("4. Cross-check against Disaggregated_Inference_Sim's closed forms")
    cm = CostModel(spec, H100_SXM)
    pre_an, dec_an = cm.prefill([T]), cm.decode([C])
    pre_tr, dec_tr = routes["dispatch (meta)"], decode_meta("llama3-8b")
    dec_fake, pre_fake = decode_fake("llama3-8b"), routes["dispatch (fake CPU tensors)"]
    # Disaggregated_Inference_Sim's closed form before its 2026-10-03 correction, which this trace prompted:
    # every step read the whole embedding table, and decode attention counted C keys, not C + 1.
    old_dec_flops = 2 * spec.matmul_params + 4 * spec.n_layers * spec.d_model * C
    old_dec_w = int(spec.weight_bytes_total)
    norms = (2 * spec.n_layers + 1) * spec.d_model * 2          # RMSNorm weights: traced, not in the closed form
    rope1 = 2 * (spec.head_dim // 2)
    table(["Quantity", "Closed form (InfSim)", "Operator trace", "Difference", "Why",
           "Closed form before 2026-10-03"], [
        ["Prefill matmul FLOPs, weights", f"{an_w:,}", f"{pre_tr.flops_of('matmul') - an_att - rope:,}", "0",
         "exact", f"{an_w:,}"],
        ["Prefill attention FLOPs", f"{2 * spec.n_layers * spec.d_model * T * (T + 1):,} (causal)", f"{an_att:,}"
         " (unmasked)", f"{an_att / (2 * spec.n_layers * spec.d_model * T * (T + 1)):.3f}x",
         "the trace counts what the kernel computes, masked scores included", "same"],
        ["Decode matmul FLOPs", f"{int(dec_an.flops):,}", f"{dec_tr.flops_of('matmul'):,}",
         f"{dec_tr.flops_of('matmul') - int(dec_an.flops):,}", f"the rotary-frequency matmul ({rope1:,}), not modelled",
         f"{old_dec_flops:,} (no self-attention: -{4 * spec.n_layers * spec.d_model:,})"],
        ["Decode weight bytes per step", f"{int(spec.weight_bytes_read(1)):,}", f"{dec_fake.weight_bytes:,}",
         f"{dec_fake.weight_bytes - int(spec.weight_bytes_read(1)):,}",
         f"the RMSNorm weights ({norms:,}), not modelled",
         f"{old_dec_w:,} (whole embedding table: +{old_dec_w - int(spec.weight_bytes_read(1)):,})"],
    ])
    agree = (dec_tr.flops_of("matmul") - int(dec_an.flops) == rope1
             and dec_fake.weight_bytes - int(spec.weight_bytes_read(1)) == norms)
    emb = spec.vocab * spec.d_model * spec.weight_bytes
    old_bytes = old_dec_w + (C + 1) * spec.kv_bytes_per_token
    old_t = cm.step_time(old_dec_flops, old_bytes)[0] - cm.step_overhead
    p(f"* **The closed form and the trace now agree** to the two terms the closed form does not model: {agree}.")
    p(f"* **Found and fixed.** This comparison found two errors in Disaggregated_Inference_Sim's closed form, corrected"
      f" there on 2026-10-03. Every decode step was charged the whole input-embedding table, {emb / 1e9:.2f} GB"
      f" ({emb / old_bytes:.1%} of the old closed form's {old_bytes / 1e9:.2f} GB step at batch 1),"
      " where a lookup reads one row; and")
    p(f"  decode attention left out the new token's attention to itself ({4 * spec.n_layers * spec.d_model:,} FLOPs"
      f" per sequence per step). On the H100 roofline the step at context {C:,} is now"
      f" {(dec_an.time - cm.step_overhead) * 1e3:.2f} ms (before step overhead), against {old_t * 1e3:.2f} ms before"
      f" the correction.")
    p()

    # ── 5. costing ──────────────────────────────────────────────────────
    h("5. Costing the traces on the H100 roofline")
    p("Per-operator roofline, summed. *Unfused*: every non-view operator reads its inputs from memory and writes its")
    p("output. *Fused*: the ideal-fusion bound, where only matmul, attention and gather operators touch memory.")
    p("The closed-form column is InfSim's CostModel step time without its fixed 0.5 ms step overhead.")
    p()
    unf, fus = device("h100"), device("h100", memory="fused")
    rows = []
    for label, tr, an in [("Prefill 2,048, meta trace (math attention)", pre_tr, pre_an),
                          ("Prefill 2,048, fake-CPU trace (fused attention)", pre_fake, pre_an),
                          ("Decode @2,048, meta trace", dec_tr, dec_an),
                          ("Decode @2,048, fake-CPU trace", dec_fake, dec_an)]:
        a, b = unf.run(tr), fus.run(tr)
        sp = a.bound_split()
        rows.append([label, ms(a.time), f"{sp['compute']:.1%}", ms(b.time), ms(an.time - cm.step_overhead),
                     an.bound])
    table(["Workload", "Unfused (ms)", "Unfused: time in compute-bound ops", "Fused bound (ms)",
           "Closed form (ms)", "Closed-form bound"], rows)
    p("Prefill is compute-bound as a whole, yet in the meta trace the materialised score matrix (bmm, softmax and")
    p("mask over 2,048 x 2,048 per head) makes half the time memory-bound: the case for flash attention, measured.")
    p()
    p("Decode traffic by operator category, per step (GB):")
    p()
    rows = []
    for label, tr in [("meta trace", dec_tr), ("fake-CPU trace", dec_fake)]:
        bc = tr.by_category()
        rows.append([label, g(tr.weight_bytes)] + [g(bc.get(c, {}).get("bytes", 0)) for c in
                                                    ["matmul", "attention", "copy", "elementwise"]] + [g(tr.bytes)])
    table(["Trace", "Weights", "matmul", "attention", "copy", "elementwise", "Total"], rows)
    clones = sum(o.bytes for o in dec_tr.ops if o.name == "aten.clone")
    cats = sum(o.bytes for o in dec_tr.ops if o.name == "aten.cat")
    p(f"In the meta trace `aten.clone` moves {g(clones)} GB per step: transformers' `repeat_kv` materialising the 8 KV")
    p(f"heads as 32 for the math attention path. `aten.cat` ({g(cats)} GB) is the DynamicCache appending one token.")
    p("The fused path (fake CPU tensors) reads the cache once through native GQA. Framework code paths, not the model,")
    p("decide these bytes.")

    # ── 6. coverage ─────────────────────────────────────────────────────
    h("6. Operator coverage: a matmul engine with a host")
    p("The hypothetical optical MAC part from Disaggregated_Inference_Sim (4,000 TFLOP/s peak at 40% efficiency,")
    p("HBM-class memory) runs only the categories listed; everything else runs on an illustrative host CPU")
    p(f"({HOST_CPU.flops_rate / 1e12:.0f} TFLOP/s, {HOST_CPU.byte_rate / 1e9:.0f} GB/s). Tensors cross the link"
      " when an operator on the other side first reads them. Fake-CPU traces (fused attention).")
    p()
    ladder = [("matmul", ["matmul"]), ("+ attention", ["matmul", "attention"]),
              ("+ elementwise", ["matmul", "attention", "elementwise"]),
              ("+ norm, reduction, softmax", ["matmul", "attention", "elementwise", "norm", "reduction", "softmax"]),
              ("+ copy, creation, gather (everything)", ["matmul", "attention", "elementwise", "norm", "reduction",
                                                          "softmax", "copy", "creation", "gather"])]
    acc = device("optical")
    rows = []
    for phase, tr in [("prefill", pre_fake), ("decode", dec_fake)]:
        for link_name in ("pcie5", "nvlink4"):
            link = LINKS[link_name]
            for label, cats_ in ladder:
                rep = Offload("x", acc, HOST_CPU, link.bandwidth, link.latency, frozenset(cats_)).run(tr)
                dc = coverage.device_coverage(rep)
                rows.append([phase, link.name, label, f"{dc['ops']:.1%}", f"{dc['flops']:.2%}", f"{dc['time']:.1%}",
                             f"{dc['host_time']:.1%}", f"{dc['transfer_time']:.1%}", ms(rep.time)])
    table(["Phase", "Link", "Accelerator runs", "Ops on accel", "FLOPs on accel", "Time on accel", "Time on host",
           "Time in transfers", "Total (ms)"], rows)
    alone = acc.run(pre_fake).time
    p(f"For reference, the same part running everything: prefill {ms(alone)} ms, "
      f"decode {ms(acc.run(dec_fake).time)} ms.")
    p("Covering 99.9% of FLOPs is not covering the time: the operators left on the host, and the traffic they cause,")
    p("dominate until nearly every category runs on the device.")
    p()
    p("Top operators by time, matmul-only engine over PCIe, prefill:")
    p()
    rep = Offload("matmul engine (optical) + host over PCIe Gen5", acc, HOST_CPU, LINKS["pcie5"].bandwidth,
                  LINKS["pcie5"].latency, frozenset(["matmul"])).run(pre_fake)
    p(coverage.markdown(rep, 8))

    # ── 7. PyTorch's own counter ─────────────────────────────────────────
    h("7. Against PyTorch's FlopCounterMode")
    m = models.build("llama3-8b")
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(models.tokens(1, T))
    meta_fc = fc.get_total_flops()
    with models.fake_cpu():
        mf = models.build("llama3-8b", device="cpu")
        with FlopCounterMode(display=False) as fc2, torch.no_grad():
            mf(models.tokens(1, T, "cpu"), use_cache=False)
    fake_fc = fc2.get_total_flops()
    table(["Trace", "FlopCounterMode", "simfront matmul", "simfront matmul + attention"], [
        ["meta (math attention)", f"{meta_fc:,}", f"{pre_tr.flops_of('matmul'):,}",
         f"{pre_tr.flops_of('matmul') + pre_tr.flops_of('attention'):,}"],
        ["fake CPU (fused attention)", f"{fake_fc:,}", f"{pre_fake.flops_of('matmul'):,}",
         f"{pre_fake.flops_of('matmul') + pre_fake.flops_of('attention'):,}"]])
    p("On the meta trace the two agree exactly. FlopCounterMode has no formula for the CPU flash-attention operator,")
    p(f"so on the fused trace it silently omits {pre_fake.flops_of('attention') / 1e12:.2f} TFLOP of attention. A")
    p("coverage report would have named it.")

    # ── 8. tests ────────────────────────────────────────────────────────
    h("8. Test suite")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:logging", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True)
    last = [ln for ln in r.stdout.splitlines() if " passed" in ln or " failed" in ln]
    p(f"`pytest`: {last[-1].strip('= ') if last else r.stdout[-200:]}")
    p()
    p(f"Whole script: {time.perf_counter() - t_start:.0f} s; peak resident memory "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:,.0f} MB.")
    (HERE / "results.md").write_text("\n".join(OUT) + "\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
