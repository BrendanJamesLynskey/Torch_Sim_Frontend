# Torch_Sim_Frontend

**`simfront`: a working front end from PyTorch and ONNX to an accelerator cost model.**
It turns a model into an operator trace four ways, costs the trace on a pluggable
accelerator model, and reports operator coverage as an engineering metric. It runs on
real published model configurations (Llama-3-8B and -70B, Mistral-7B, Qwen2.5-0.5B,
GPT-2) **without downloading or allocating a single weight**, on a CPU-only machine.

It is the companion code for deck 10 (from PyTorch and ONNX to an accelerator model) of
the [Simulation Engineering Toolkit](https://github.com/BrendanJamesLynskey/SimEng_Hub_Toolkit)
series, and it starts from the roofline of
[Disaggregated_Inference_Sim](https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim)
(LLM Inference Simulators).

| Front end | How | What it sees |
|---|---|---|
| **Dispatch trace** | A `TorchDispatchMode` records every ATen call, on the meta device or under fake CPU tensors | What PyTorch would execute on that device, call by call |
| **`torch.export`** | Export, `run_decompositions()` to Core ATen, walk the FX graph | One whole-program graph in a small operator set |
| **`torch.compile` backend** | A Dynamo backend that lowers each graph with AOTAutograd, costs it, then runs it | The graphs a user's unchanged program produces, graph breaks included |
| **ONNX** | Export without weights, `infer_shapes(data_prop=True)`, walk the graph | The framework-neutral route; folds constant subgraphs as a runtime would |

All four produce the same trace format (`simfront-trace/1`), with every tensor's shape,
element size and identity, and weights told apart from activations. That is what makes
them comparable, and what lets a cost model follow data between devices.

**Why trust it.** All numbers below come from [`examples/results.md`](examples/results.md).

* **The four front ends agree exactly.** On Llama-3-8B (2,048-token prefill, meta
  device) all four count 32,938,104,455,168 matmul FLOPs. ONNX is lower by exactly
  262,144: it constant-folds the rotary-frequency matmul. All four count
  15,026,626,560 bytes of weights read.
* **The trace equals the closed form.** The weight matmuls equal Disaggregated_Inference_Sim's
  2 × parameters × tokens exactly. A Hypothesis property test checks the same identity for
  random Llama-shaped configurations, batch sizes and lengths.
* **It matches PyTorch's own counter.** On the meta trace, `torch.utils.flop_counter.FlopCounterMode`
  gives the same total exactly.
* **It checks the tests themselves.** Fake tensors trace exactly what real tensors run,
  operator by operator, on a model without data-dependent Python. ONNX Runtime runs the
  exported model and matches PyTorch to 1e-4.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only is enough
pip install -e ".[test]"
pytest -p no:logging                                    # 41 tests, about a minute
simfront --model llama3-70b --tokens 2048               # 70B parameters, no memory used
simfront --model llama3-8b --decode 2048 --fake-cpu     # one decode step, fused attention
simfront --model llama3-8b --offload optical            # a matmul-only engine plus a host
python examples/results.py                              # regenerates examples/results.md
```

```python
from simfront import device, models
from simfront.capture import trace_dispatch

m = models.build("llama3-8b")                             # meta device: shapes, no storage
trace, _ = trace_dispatch(m, models.tokens(1, 2048))
print(len(trace.ops), trace.flops, trace.weight_bytes)    # 3557 operators
report = device("h100").run(trace)                        # per-operator roofline
print(report.time, report.bound_split())
```

## Selected results

**Real models, no weights** (§2). The capture time is for a 2,048-token prefill on this
machine's CPU:

| Model | Parameters | Operators | Capture (s) |
|---|---|---|---|
| Llama-3-8B | 8.03 B | 3,557 | 0.64 |
| Llama-3-70B | 70.55 B | 8,837 | 1.23 |
| Mistral-7B | 7.24 B | 3,685 | 0.66 |

**Checking the closed form found two errors in it, since fixed** (§4). Disaggregated_Inference_Sim's
decode step charged the whole embedding table (1.05 GB, 6.4% of a Llama-3-8B decode
step's bytes) on every token, where a lookup reads one row. Its decode FLOPs also left out
the new token attending to itself (524,288 FLOPs). Neither is large, and both are the
kind of thing only an independent count finds. Both were corrected in
Disaggregated_Inference_Sim (and its JavaScript and Rust ports) on 2026-10-03. The closed form
and the trace now agree exactly, apart from two small terms the closed form does not model: the
rotary-frequency matmul (128 FLOPs) and the RMSNorm weights (532,480 bytes). The tests that
reported the discrepancy now check this agreement. The corrected decode step (5.70 ms at
context 2,048) equals the trace's ideal-fusion bound.

**What the trace shows depends on how it was captured** (§3, §5):

* On the **meta device**, attention has no fused kernel and decomposes into `bmm`,
  `_safe_softmax` and a mask over the full 2,048 × 2,048 score matrix per head. On the
  H100 roofline that prefill takes 126.68 ms, with only 44.6% of its time in
  compute-bound operators.
* Under **fake CPU tensors**, attention is one fused flash-attention operator per layer,
  and the same prefill takes 76.14 ms. The ideal-fusion bound is 60.62 ms; the closed
  form gives 58.53 ms.
* In **decode**, the meta trace moves 25.02 GB per step against 15.84 GB for the fused
  trace. 4.30 GB of the difference is transformers' `repeat_kv` materialising 8 KV
  heads as 32.
* **transformers itself changes path when it detects tracing.** In eager mode on real
  data it inspects the mask with `.item()` and uses SDPA's native GQA. When it detects
  tracing it builds an explicit mask and repeats the KV heads. The arithmetic is the
  same; the memory traffic is not. A test pins this down.

**Operator coverage is a time metric, not a count** (§6). Take a hypothetical matmul engine
(Disaggregated_Inference_Sim's optical part) attached to a host CPU over PCIe Gen5:

| Engine runs | FLOPs on the engine | Time on the engine | Prefill (ms) |
|---|---|---|---|
| matmul | 93.25% | 1.3% | 1,514.34 |
| + attention | 99.98% | 4.8% | 426.93 |
| + elementwise | 100.00% | 10.9% | 288.57 |
| everything | 100.00% | 100.0% | 36.16 |

In decode, adding attention without the KV-cache append (`cat`) makes the step *slower*
(16.59 → 19.86 ms), because the cache then crosses the link twice per step.

**PyTorch's FLOP counter misses fused CPU attention** (§7). `FlopCounterMode` has no
formula for `_scaled_dot_product_flash_attention_for_cpu`, so on the fused trace it
silently omits 2.22 TFLOP. simfront's coverage report names every operator without a
cost rule rather than costing it at zero. On its first run it named two missing rules
for GPT-2's ONNX export (`SplitToSequence`, `SequenceAt`), which were then added.

## How it works

| File | What |
|------|------|
| [`trace.py`](src/simfront/trace.py) | `TensorMeta`, `Op`, `Trace`: the format every front end produces, with JSON I/O |
| [`rules.py`](src/simfront/rules.py) | FLOPs and bytes per operator, for ATen and ONNX, by category (matmul, attention, elementwise, softmax, copy, view, ...) |
| [`capture/dispatch.py`](src/simfront/capture/dispatch.py) | `OpTrace`, a `TorchDispatchMode` that follows tensor identity and propagates "is a weight" through views |
| [`capture/export.py`](src/simfront/capture/export.py), [`compile.py`](src/simfront/capture/compile.py), [`fx_walk.py`](src/simfront/capture/fx_walk.py) | `torch.export` and the `SimBackend` for `torch.compile`, sharing one FX-graph walker |
| [`capture/onnx_walk.py`](src/simfront/capture/onnx_walk.py) | Weightless ONNX export; graph walk with data-propagating shape inference and constant folding |
| [`models.py`](src/simfront/models.py) | Published configurations (in `configs/`, each with its source URL); meta-device and fake-CPU builds; the link to InfSim's `ModelSpec` |
| [`cost.py`](src/simfront/cost.py) | `Roofline` (unfused and ideal-fusion bounds) and `Offload` (an accelerator for some categories, a host, a link) |
| [`coverage.py`](src/simfront/coverage.py) | Cost-rule coverage and device coverage (by operators, FLOPs and time), as a Markdown report |

**Counting conventions, stated so they can be argued with.**

* Attention FLOPs are counted *unmasked*. A causal kernel that skips masked blocks does
  about half.
* Every non-view operator reads its inputs from memory and writes its outputs (the
  unfused bound). `memory="fused"` gives the opposite bound, where only matmul,
  attention and gather operators touch memory.
* Elementwise operations count one FLOP per output element; softmax and norms count five.
* An embedding lookup reads the rows it gathers, not the table.

## Specification, test plan and traceability

[`docs/spec.md`](docs/spec.md) states 17 requirements in EARS patterns (ubiquitous,
event-driven, state-driven, unwanted behaviour, optional feature), each with one
verification method. Tests name the requirements they verify (`@pytest.mark.req("SF-04")`).
[`ci/trace_matrix.py`](ci/trace_matrix.py) runs the suite and writes
[`docs/traceability.md`](docs/traceability.md), checking both directions: every
requirement has a passing test, and every test traces to a requirement or is listed as
untraced.

Its first run found a real gap. SF-15 ("if a library upgrade changes the traced
arithmetic, CI shall fail") was claimed by the CI gate, but no test checked that the gate
actually fails. [`tests/test_gate.py`](tests/test_gate.py) now does.
[`docs/test_plan.md`](docs/test_plan.md) sets out the oracles, criteria and deliverables.

## Gotchas found on the way

* **The meta device and transformers' masks.** With the cache off, transformers 5 calls
  `.item()` on a data-dependent tensor to detect packed sequences. The meta device cannot
  do that, so meta prefill traces keep the cache on. Under fake tensors, export or
  compile, transformers takes its tracing path and the cache can be off.
* **`DynamicCache` is not a pytree.** `torch.export` and `torch.compile` cannot return it,
  so those routes trace prefill with `use_cache=False`. Decode steps are traced with the
  dispatch route.
* **ONNX export without weights.** The exporter's optimiser needs real weight values and
  renames the weights it pre-transposes, so it is turned off. `trace_onnx` folds constant
  subgraphs itself.
* **ONNX shapes need values.** With static shapes, `infer_shapes` without `data_prop=True`
  leaves every attention MatMul of an exported Llama without dimensions.
* **ONNX does not tell weights from buffers.** Both are initializers, so `export_onnx`
  returns the parameter names, including tied aliases (`named_parameters(remove_duplicate=False)`;
  GPT-2's `lm_head.weight` is `transformer.wte.weight`).

## Not modelled

* Training graphs. The `torch.compile` route sees backward graphs through AOTAutograd,
  but only forward graphs are costed here.
* Mixture-of-experts routing. Data-dependent expert selection needs values, which the
  meta device does not have.
* Collective communication and tensor parallelism. A trace is one device's work.
* Real kernels' efficiency per shape. The roofline applies one derating per device. A
  calibrated per-operator table (as in deck 06 of LLM Inference Simulators) is the next
  step.

## CI

* **GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):
  * ruff;
  * the test suite on Python 3.10 and 3.12 with CPU-only PyTorch;
  * the arithmetic gate;
  * a CLI smoke test that traces Llama-3-70B.
* **Jenkins** ([`Jenkinsfile`](Jenkinsfile)):
  * lint;
  * tests with JUnit and Cobertura coverage;
  * the arithmetic, framework-drift and speed gate against
    [`ci/perf_baseline.json`](ci/perf_baseline.json). FLOPs and weight bytes must not
    change; operator counts and bytes moved are reported when a library upgrade changes them;
  * the traceability matrix;
  * `results.md`;
  * a nightly sweep over models, phases and lengths.

## References

* [PyTorch: `torch.export`](https://docs.pytorch.org/docs/stable/export.html),
  [custom `torch.compile` backends](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html),
  [the meta device](https://docs.pytorch.org/docs/stable/meta.html) and
  [`__torch_dispatch__`](https://docs.pytorch.org/docs/stable/notes/extending.html).
* [ONNX shape inference](https://onnx.ai/onnx/api/shape_inference.html) and the
  [ONNX exporter](https://docs.pytorch.org/docs/stable/onnx_export.html).
* S. Williams, A. Waterman, D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 52(4), 2009.
* T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022 ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)).

## How the measurements are made

The tools and methods this repository measures with are explained, with their overheads, accuracy and pitfalls, in [SimEng 12: Measurement Tools and Methods](https://brendanjameslynskey.github.io/SimEng_12_Measurement_Tools_and_Methods/) and the series glossaries:

* [PyTorch's FLOP counter (FlopCounterMode)](https://brendanjameslynskey.github.io/SimEng_12_Measurement_Tools_and_Methods/#card-flopcounter)
* [roofline costing](https://brendanjameslynskey.github.io/LLM_Hub_Inference_Simulators/#g-roofline)

## Related

* [Disaggregated_Inference_Sim](https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim): the simulator whose devices and closed forms this uses.
* [InfSim 09: Framework Integration](https://brendanjameslynskey.github.io/InfSim_09_Framework_Integration/): the integration routes, explained.
* [Simulation Engineering Toolkit](https://github.com/BrendanJamesLynskey/SimEng_Hub_Toolkit): the series this belongs to.

## Licence

MIT.
