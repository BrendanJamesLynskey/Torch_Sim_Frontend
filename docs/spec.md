# simfront: requirements specification

Version 1.1, for simfront 0.1. A worked example for deck SimEng 09 (specifications,
requirements and test plans): every requirement below is written in an EARS pattern,
has a single verification method, and is traced to the tests or CI gate that verify it
in [`traceability.md`](traceability.md), which `ci/trace_matrix.py` generates from the
test run itself.

## 1. Scope

simfront turns PyTorch and ONNX models into operator traces and costs them on an
accelerator model. It is a design-exploration tool: its output informs architecture
decisions and is never shipped on a product, so it is not mission-mode software (see
the deck for the distinction). Its users are architects and simulator developers.

## 2. Definitions

* **Trace**: the ordered list of operators captured from one forward pass.
* **Front end**: one of the four capture routes (dispatch, export, compile, ONNX).
* **Weight bytes**: bytes read from model parameters, as opposed to activations or buffers.
* **Closed form**: the analytic FLOP and byte counts of Disaggregated_Inference_Sim's `CostModel`.
* **Reference models**: the configurations in `src/simfront/configs/`.

Verification methods: **T** test, **A** analysis, **I** inspection, **D** demonstration.

## 3. Requirements

| ID | Type | Pattern | Requirement | Verification |
|---|---|---|---|---|
| SF-01 | Functional | Ubiquitous | The front end shall record, for every captured operator, the shape, element size and identity of each input and output tensor and whether it is a model parameter. | T |
| SF-02 | Functional | Event-driven | When a model is traced on the meta device, the front end shall allocate no parameter storage. | T |
| SF-03 | Functional | Ubiquitous | For SwiGLU decoder configurations, the traced weight-matmul FLOPs of a prefill shall equal 2 × matmul parameters × tokens exactly. | T |
| SF-04 | Functional | Ubiquitous | The four front ends shall report identical matmul FLOPs and weight bytes for the same model and input, apart from constant folding that is documented per case. | T |
| SF-05 | Functional | Event-driven | When a decode step is traced, the traced matmul FLOPs shall equal the closed form plus the self-attention term (4 × layers × d_model per token). | T |
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

## 4. Assumptions and constraints

* PyTorch 2.6 or later, transformers 4.50 or later; tested with the versions in `results.md` §1.
* Batch-1 static shapes; dynamic shapes are out of scope for version 1.0.
* Cost-model numbers are as accurate as the device parameters they are given; these come
  from Disaggregated_Inference_Sim and are datasheet-level or illustrative.

## 5. Change history

| Version | Change |
|---|---|
| 1.0 | First issue. |
| 1.1 | SF-15's verification changed from "T (CI gate)" to T: the first traceability run showed that no test checked the gate itself fails; `tests/test_gate.py` added. |
