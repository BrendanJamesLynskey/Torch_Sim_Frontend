# simfront: requirements specification

Version 1.3, for simfront 0.1. A worked example for deck SimEng 09 (specifications,
requirements and test plans): every requirement below is written in an EARS pattern,
has a single verification method, and is traced to the tests or CI gate that verify it
in [`traceability.md`](traceability.md), which `ci/trace_matrix.py` generates from the
test run itself.

## 1. Scope

simfront turns PyTorch and ONNX models into operator traces and costs them on an
accelerator model: an analytic roofline (`cost.py`), or an event-driven SimPy model of a tiled
accelerator (`simfront.accel`, specified as a hardware block in [`accel_spec.md`](accel_spec.md)). It is a design-exploration tool: its output informs architecture
decisions and is never shipped on a product, so it is not mission-mode software (see
the deck for the distinction). Its users are architects and simulator developers.

## 2. Definitions

* **Trace**: the ordered list of operators captured from one forward pass.
* **Front end**: one of the four capture routes (dispatch, export, compile, ONNX).
* **Weight bytes**: bytes read from model parameters, as opposed to activations or buffers.
* **Closed form**: the analytic FLOP and byte counts of Disaggregated_Inference_Sim's `CostModel`.
* **Reference models**: the configurations in `src/simfront/configs/`.
* **Tile**: the unit the accelerator model loads, computes and stores (`simfront.accel.lower`).
* **Program**: a trace lowered to tiles for one `AccelConfig`; every engine runs the same program.
* **Engines**: the SimPy model (reference), the fast path (Python recurrence and C++ module) and the
  cycle-stepped twin.

Verification methods: **T** test, **A** analysis, **I** inspection, **D** demonstration.

## 3. Requirements

| ID | Type | Pattern | Requirement | Verification |
|---|---|---|---|---|
| SF-01 | Functional | Ubiquitous | The front end shall record, for every captured operator, the shape, element size and identity of each input and output tensor and whether it is a model parameter. | T |
| SF-02 | Functional | Event-driven | When a model is traced on the meta device, the front end shall allocate no parameter storage. | T |
| SF-03 | Functional | Ubiquitous | For SwiGLU decoder configurations, the traced weight-matmul FLOPs of a prefill shall equal 2 × matmul parameters × tokens exactly. | T |
| SF-04 | Functional | Ubiquitous | The four front ends shall report identical matmul FLOPs and weight bytes for the same model and input, apart from constant folding that is documented per case. | T |
| SF-05 | Functional | Event-driven | When a decode step is traced, the traced matmul FLOPs shall equal the closed form plus the rotary-frequency matmul, and the traced weight bytes shall equal the closed form's per-step weight traffic plus the RMSNorm weights. | T |
| SF-06 | Functional | Unwanted behaviour | If an operator has no cost rule, then the front end shall cost it at zero and name it in the coverage report. | T |
| SF-07 | Functional | Event-driven | When a model is exported to ONNX without weights, the front end shall recover the shape and parameter identity of every weight. | T |
| SF-08 | Functional | State-driven | While the `torch.compile` backend is active, the user's program shall produce outputs identical to eager execution. | T |
| SF-09 | Functional | Ubiquitous | The roofline cost model shall use the same achievable FLOP and byte rates as Disaggregated_Inference_Sim's `CostModel` for the same device. | T |
| SF-10 | Functional | Optional feature | Where the offload cost model is selected, the cost model shall charge a link transfer the first time an operator reads a tensor resident on the other side, and not again. | T |
| SF-11 | Functional | Event-driven | When a model without data-dependent Python control flow is traced under fake tensors, the front end shall record the same operator sequence and shapes as a run with real tensors. | T |
| SF-12 | Functional | Ubiquitous | The front end shall report operator coverage by operator count, FLOPs and time. | T |
| SF-13 | Non-functional (performance) | Ubiquitous | The dispatch front end shall capture a Llama-3-8B 2,048-token prefill trace in less than 5 s on the CI machine. | T |
| SF-14 | Non-functional (resource) | Ubiquitous | The front end shall trace every reference model with a peak resident memory below 2 GB. | D (`results.md`) |
| SF-15 | Non-functional (maintainability) | Unwanted behaviour | If a library upgrade changes the traced FLOPs or weight bytes of a reference trace, then CI shall fail. | T |
| SF-16 | Non-functional (portability) | Ubiquitous | The front end shall run on a CPU-only machine with no GPU and no model weights. | D (GitHub Actions) |
| SF-17 | Functional | Ubiquitous | The cost rules shall count each operator's FLOPs and bytes by the conventions stated in `rules.py` and `cost.py`. | T |
| SF-18 | Functional | Ubiquitous | The accelerator model shall lower every operator that has a cost rule into tiles that fit the tile budget, and the tiles of a GEMM shall perform exactly its multiply-accumulates. | T |
| SF-19 | Functional | State-driven | While the on-chip buffer cannot hold the next tile, the load DMA shall not start its transfer, and buffer occupancy shall never exceed its capacity. | T |
| SF-20 | Functional | Event-driven | When an operator reads an activation, its first load shall not start before the operator that produced it has stored its results. | T |
| SF-21 | Functional | Ubiquitous | The accelerator model shall report latency, utilisation per component, a stall breakdown that sums to the latency, a hot-spot, a per-operator latency histogram, a timeline plot and a Chrome trace. | T |
| SF-22 | Functional | State-driven | While the two DMA engines cannot contend for a memory channel, the fast path (Python and C++) shall produce per-tile timings bit-identical to the SimPy model's. | T |
| SF-23 | Functional | Unwanted behaviour | If the configuration lets the DMA engines contend for one memory channel, then the fast path shall refuse to run. | T |
| SF-24 | Functional | Optional feature | Where durations are quantised to whole cycles, the cycle-stepped twin shall produce per-tile timings identical to the event-driven model's. | T |
| SF-25 | Functional | Ubiquitous | The NTT polynomial product shall equal schoolbook multiplication in Z_q[X]/(X^N + 1). | T |
| SF-26 | Functional | Optional feature | Where an in-transit stage is configured, the model shall run NTT operators in the read path, never faster than the stage's operations-per-byte budget allows. | T |
| SF-27 | Functional | Ubiquitous | The FIFO model shall satisfy Little's law (time-averaged depth = throughput x mean time in the FIFO) on every run, with depth never above capacity. | T |
| SF-28 | Functional | Ubiquitous | The execution-provider partitioner shall assign every ONNX node exactly once, and the ONNX Runtime profile shall name the provider of every node run. | T |

## 4. Assumptions and constraints

* PyTorch 2.6 or later, transformers 4.50 or later; tested with the versions in `results.md` §1.
* Batch-1 static shapes; dynamic shapes are out of scope for version 1.0.
* Cost-model numbers are as accurate as the device parameters they are given; these come
  from Disaggregated_Inference_Sim and are datasheet-level or illustrative.
* The accelerator presets (`simfront.accel.hw`) are illustrative; what the accelerator model
  leaves out is listed in [`accel_spec.md`](accel_spec.md) §8.
* The C++ fast path needs a C++20 compiler at install time; without one, the Python recurrence
  is used and SF-22 is verified for it alone (CI requires the C++ module).

## 5. Change history

| Version | Change |
|---|---|
| 1.0 | First issue. |
| 1.1 | SF-15's verification changed from "T (CI gate)" to T: the first traceability run showed that no test checked the gate itself fails; `tests/test_gate.py` added. |
| 1.2 | SF-05 reworded (2026-10-03): Disaggregated_Inference_Sim's closed form was corrected (self-attention term, embedding rows instead of the whole table), so the requirement now states agreement, and adds weight bytes. |
| 1.3 | SF-18 to SF-28 added (2026-10-05) for the SimPy accelerator model (`simfront.accel`): its fast path, cycle-stepped twin, FIFO, NTT workload and execution-provider partitioning. |
