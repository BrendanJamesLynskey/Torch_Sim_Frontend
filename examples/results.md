# Torch_Sim_Frontend: recorded results

Written by `examples/results.py`. Every number in the README and in deck SimEng 10 comes from this file.
Models are built from published configurations on the meta device or under fake tensors: no weights are
downloaded or allocated. Prefill is one 2,048-token prompt; decode is one step after a 2,048-token context;
batch 1, BF16. Device rates are Disaggregated_Inference_Sim's H100-SXM roofline (peak x efficiency).

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.14.1+cpu |
| transformers | 5.18.0 |
| onnx | 1.23.1 |
| CPU | x86_64 |
| simfront commit | e8ef627 |


## 2. Real model configurations traced without weights

Dispatch trace on the meta device, one 2,048-token prefill. Parameters are counted from the instantiated
model; every parameter is a meta tensor (checked), so none of this used parameter memory.

| Model | Parameters | Operators | Op types | Capture (s) | TFLOP | Weights read (GB) | Types with a cost rule |
|---|---|---|---|---|---|---|---|
| llama3-8b | 8.03 B | 3,557 | 30 | 0.62 | 32.97 | 15.03 | 30 |
| llama3-70b | 70.55 B | 8,837 | 30 | 1.21 | 295.83 | 139.04 | 30 |
| mistral-7b | 7.24 B | 3,685 | 31 | 0.65 | 31.36 | 14.24 | 31 |
| qwen2.5-0.5b | 0.49 B | 2,677 | 31 | 0.40 | 2.39 | 0.99 | 31 |
| gpt2 | 0.12 B | 771 | 27 | 0.18 | 0.67 | 0.25 | 27 |

Configurations: the architectural fields of the published Hugging Face `config.json` files (sources in
`src/simfront/configs/`). Llama-3 configs are from the NousResearch mirror, which is not gated.

## 3. Four front ends, one model: Llama-3-8B prefill

The same model, on the meta device unless stated, captured four ways. FLOPs are the trace's total; matmul
FLOPs include the attention matmuls (QK^T, AV) where attention is decomposed.

| Front end | Operators | Op types | Matmul FLOPs | Fused-attention FLOPs | Total TFLOP | Bytes moved (GB) | Weights read (GB) | Capture (s) |
|---|---|---|---|---|---|---|---|---|
| dispatch (meta) | 3,557 | 30 | 32,938,104,455,168 | 0 | 32.970 | 214.35 | 15.027 | 0.5 |
| torch.export + Core ATen | 3,850 | 37 | 32,938,104,455,168 | 0 | 32.988 | 333.97 | 15.027 | 13.4 |
| torch.compile backend | 3,426 | 35 | 32,938,104,455,168 | 0 | 32.970 | 187.40 | 15.027 | 12.0 |
| ONNX (exported without weights) | 6,007 | 35 | 32,938,104,193,024 | 0 | 32.980 | 169.10 | 15.027 | 0.7 |
| dispatch (fake CPU tensors) | 4,577 | 37 | 30,739,081,199,616 | 2,220,498,092,032 | 32.966 | 70.09 | 15.027 | 0.9 |

* The four meta-device routes agree exactly on matmul FLOPs, 32,938,104,455,168 (ONNX: 32,938,104,193,024, the rotary-frequency matmul of 262,144 FLOPs constant-folded),
  and on weights read, 15,026,626,560 bytes.
* That is 2 x matmul parameters x tokens (30,739,080,937,472) + unmasked attention 4 x layers x d x T^2 (2,199,023,255,552) + the rotary matmul (262,144).
* torch.compile captured 1 graph(s), no graph breaks. The ONNX export (without weights) took 14.9 s and is 11.3 MB.
* Under fake CPU tensors attention is one fused operator per layer (`_scaled_dot_product_flash_attention_for_cpu`); its FLOPs include the softmax (5 per score).

Bytes moved, by operator category (GB):

| Front end | matmul | attention | softmax | elementwise | copy | reduction | norm | gather | creation |
|---|---|---|---|---|---|---|---|---|---|
| dispatch (meta) | 64.94 | 0.00 | 34.36 | 68.83 | 43.59 | 2.18 | 0.00 | 0.03 | 0.40 |
| torch.export + Core ATen | 64.94 | 0.00 | 34.36 | 164.97 | 28.83 | 6.48 | 0.00 | 0.03 | 34.36 |
| torch.compile backend | 64.94 | 0.00 | 34.36 | 68.32 | 17.55 | 2.18 | 0.00 | 0.03 | 0.00 |
| ONNX (exported without weights) | 45.62 | 0.00 | 17.18 | 88.55 | 15.54 | 2.18 | 0.00 | 0.03 | 0.00 |
| dispatch (fake CPU tensors) | 26.29 | 2.15 | 0.00 | 29.40 | 10.04 | 2.18 | 0.00 | 0.03 | 0.00 |

ATen views are free. ONNX Transpose, Expand and Slice materialise and are counted as copies; the ATen
decompositions differ in where they upcast to FP32 and copy. The arithmetic agrees; the memory traffic is a
property of the decomposition, not of the model.

## 4. Cross-check against Disaggregated_Inference_Sim's closed forms

| Quantity | Closed form (InfSim) | Operator trace | Difference | Why |
|---|---|---|---|---|
| Prefill matmul FLOPs, weights | 30,739,080,937,472 | 30,739,080,937,472 | 0 | exact |
| Prefill attention FLOPs | 1,100,048,498,688 (causal) | 2,199,023,255,552 (unmasked) | 1.999x | the trace counts what the kernel computes, masked scores included |
| Decode matmul FLOPs | 16,083,058,688 | 16,083,583,104 | 524,416 | the new token also attends to itself: 4 x layers x d = 524,288, + rotary 128 |
| Decode weight bytes per step | 16,059,990,016 | 15,009,857,536 | 1,050,132,480 | the closed form reads the whole embedding table; a lookup reads one row |

The embedding-table term is 1.05 GB per decode step, 6.4% of the closed form's 16.33 GB: on the H100 roofline, 0.39 ms of a 6.09 ms step (before step overhead). The operator trace corrects
the closed form, which is the point of checking one against the other.

## 5. Costing the traces on the H100 roofline

Per-operator roofline, summed. *Unfused*: every non-view operator reads its inputs from memory and writes its
output. *Fused*: the ideal-fusion bound, where only matmul, attention and gather operators touch memory.
The closed-form column is InfSim's CostModel step time without its fixed 0.5 ms step overhead.

| Workload | Unfused (ms) | Unfused: time in compute-bound ops | Fused bound (ms) | Closed form (ms) | Closed-form bound |
|---|---|---|---|---|---|
| Prefill 2,048, meta trace (math attention) | 126.68 | 44.6% | 71.01 | 58.53 | compute |
| Prefill 2,048, fake-CPU trace (fused attention) | 76.14 | 79.6% | 60.62 | 58.53 | compute |
| Decode @2,048, meta trace | 9.34 | 0.0% | 6.41 | 6.09 | memory |
| Decode @2,048, fake-CPU trace | 5.91 | 0.0% | 5.70 | 6.09 | memory |

Prefill is compute-bound as a whole, yet in the meta trace the materialised score matrix (bmm, softmax and
mask over 2,048 x 2,048 per head) makes half the time memory-bound: the case for flash attention, measured.

Decode traffic by operator category, per step (GB):

| Trace | Weights | matmul | attention | copy | elementwise | Total |
|---|---|---|---|---|---|---|
| meta trace | 15.01 | 17.18 | 0.00 | 5.66 | 2.16 | 25.02 |
| fake-CPU trace | 15.01 | 15.01 | 0.27 | 0.54 | 0.01 | 15.84 |

In the meta trace `aten.clone` moves 4.30 GB per step: transformers' `repeat_kv` materialising the 8 KV
heads as 32 for the math attention path. `aten.cat` (0.54 GB) is the DynamicCache appending one token.
The fused path (fake CPU tensors) reads the cache once through native GQA. Framework code paths, not the model,
decide these bytes.

## 6. Operator coverage: a matmul engine with a host

The hypothetical optical MAC part from Disaggregated_Inference_Sim (4,000 TFLOP/s peak at 40% efficiency,
HBM-class memory) runs only the categories listed; everything else runs on an illustrative host CPU
(2 TFLOP/s, 200 GB/s). Tensors cross the link when an operator on the other side first reads them. Fake-CPU traces (fused attention).

| Phase | Link | Accelerator runs | Ops on accel | FLOPs on accel | Time on accel | Time on host | Time in transfers | Total (ms) |
|---|---|---|---|---|---|---|---|---|
| prefill | PCIe Gen5 x16 | matmul | 16.0% | 93.25% | 1.3% | 87.1% | 11.7% | 1,514.34 |
| prefill | PCIe Gen5 x16 | + attention | 18.2% | 99.98% | 4.8% | 48.8% | 46.4% | 426.93 |
| prefill | PCIe Gen5 x16 | + elementwise | 71.6% | 100.00% | 10.9% | 21.2% | 67.8% | 288.57 |
| prefill | PCIe Gen5 x16 | + norm, reduction, softmax | 76.3% | 100.00% | 13.3% | 20.7% | 66.0% | 243.74 |
| prefill | PCIe Gen5 x16 | + copy, creation, gather (everything) | 100.0% | 100.00% | 100.0% | 0.0% | 0.0% | 36.16 |
| prefill | NVLink 4 (one direction) | matmul | 16.0% | 93.25% | 1.4% | 96.6% | 2.0% | 1,364.80 |
| prefill | NVLink 4 (one direction) | + attention | 18.2% | 99.98% | 7.9% | 80.3% | 11.8% | 259.40 |
| prefill | NVLink 4 (one direction) | + elementwise | 71.6% | 100.00% | 25.4% | 49.3% | 25.3% | 124.36 |
| prefill | NVLink 4 (one direction) | + norm, reduction, softmax | 76.3% | 100.00% | 29.8% | 46.3% | 23.9% | 108.77 |
| prefill | NVLink 4 (one direction) | + copy, creation, gather (everything) | 100.0% | 100.00% | 100.0% | 0.0% | 0.0% | 36.16 |
| decode | PCIe Gen5 x16 | matmul | 17.4% | 93.24% | 33.8% | 24.9% | 41.3% | 16.59 |
| decode | PCIe Gen5 x16 | + attention | 19.8% | 99.98% | 28.7% | 14.0% | 57.3% | 19.86 |
| decode | PCIe Gen5 x16 | + elementwise | 74.7% | 100.00% | 27.0% | 12.8% | 60.1% | 21.12 |
| decode | PCIe Gen5 x16 | + norm, reduction, softmax | 79.7% | 100.00% | 27.9% | 13.2% | 58.9% | 20.45 |
| decode | PCIe Gen5 x16 | + copy, creation, gather (everything) | 100.0% | 100.00% | 100.0% | 0.0% | 0.0% | 5.91 |
| decode | NVLink 4 (one direction) | matmul | 17.4% | 93.24% | 43.4% | 32.0% | 24.7% | 12.92 |
| decode | NVLink 4 (one direction) | + attention | 19.8% | 99.98% | 45.3% | 22.1% | 32.6% | 12.59 |
| decode | NVLink 4 (one direction) | + elementwise | 74.7% | 100.00% | 41.2% | 19.6% | 39.2% | 13.85 |
| decode | NVLink 4 (one direction) | + norm, reduction, softmax | 79.7% | 100.00% | 43.3% | 20.5% | 36.2% | 13.20 |
| decode | NVLink 4 (one direction) | + copy, creation, gather (everything) | 100.0% | 100.00% | 100.0% | 0.0% | 0.0% | 5.91 |

For reference, the same part running everything: prefill 36.16 ms, decode 5.91 ms.
Covering 99.9% of FLOPs is not covering the time: the operators left on the host, and the traffic they cause,
dominate until nearly every category runs on the device.

Top operators by time, matmul-only engine over PCIe, prefill:

Operator coverage: llama3-8b, dispatch trace, on matmul engine (optical) + host over PCIe Gen5

* cost rules: 37/37 operator types, 4577/4577 operator calls
* without a rule (costed at zero): none
* on the accelerator: 16.0% of costed ops, 93.25% of FLOPs, 1.3% of time (host 87.1%, transfers 11.7%)

| Operator | Category | Calls | GFLOP | MB moved | Time share | Unit |
|---|---|---|---|---|---|---|
| `aten._scaled_dot_product_flash_attention_for_cpu` | attention | 32 | 2,220.5 | 2,147.5 | 73.32% | host |
| `aten.mul` | elementwise | 292 | 2.7 | 14,937.0 | 7.59% | host |
| `aten.mm` | matmul | 225 | 30,739.1 | 26,288.8 | 6.62% | accel |
| `aten.silu` | elementwise | 32 | 0.9 | 3,758.1 | 3.19% | host |
| `aten.add` | elementwise | 196 | 0.9 | 5,235.7 | 2.86% | host |
| `aten._to_copy` | copy | 133 | 0.0 | 6,546.3 | 2.16% | host |
| `aten.pow` | elementwise | 65 | 0.5 | 4,362.1 | 1.44% | host |
| `aten.clone` | copy | 64 | 0.0 | 2,147.5 | 1.27% | host |

## 7. Against PyTorch's FlopCounterMode

| Trace | FlopCounterMode | simfront matmul | simfront matmul + attention |
|---|---|---|---|
| meta (math attention) | 32,938,104,455,168 | 32,938,104,455,168 | 32,938,104,455,168 |
| fake CPU (fused attention) | 30,739,081,199,616 | 30,739,081,199,616 | 32,959,579,291,648 |

On the meta trace the two agree exactly. FlopCounterMode has no formula for the CPU flash-attention operator,
so on the fused trace it silently omits 2.22 TFLOP of attention. A
coverage report would have named it.

## 8. Test suite

`pytest`: 38 passed in 37.77s

Whole script: 90 s; peak resident memory 647 MB.
