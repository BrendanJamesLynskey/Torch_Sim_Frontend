# simfront: test plan

Version 1.0. A worked example for deck SimEng 09. Its sections follow the usual
test-plan contents (as in ISO/IEC/IEEE 29119-3), cut to what a small tool needs.

## 1. Test items and scope

simfront 0.1: the four front ends (`simfront.capture`), the cost rules (`rules.py`), the
cost models (`cost.py`), coverage reporting (`coverage.py`) and the CLI, against the
requirements in [`spec.md`](spec.md). **Out of scope:** the accuracy of the device
parameters (owned by Disaggregated_Inference_Sim), and PyTorch and transformers themselves.

## 2. Approach: what tells us a result is right

A test is only as good as its oracle. Each level uses an oracle that does not share code
with what it checks:

| Level | What is tested | Oracle | Requirements |
|---|---|---|---|
| Unit | Each cost rule | FLOP and byte counts worked by hand | SF-06, SF-17 |
| Unit (property) | Rules and cost models for any input | Invariants: 2MKN; faster hardware is never slower | SF-17 |
| Integration | Each front end on real configurations | Disaggregated_Inference_Sim's closed forms; PyTorch's `FlopCounterMode` | SF-02, SF-03, SF-05 |
| Integration (property) | Random Llama-shaped configs, lengths, batches | The closed form, exactly (Hypothesis) | SF-03 |
| Differential | The four front ends against each other | Each other: they must agree | SF-04, SF-07 |
| Faithfulness | Fake tensors and the compile backend | A real run with data; eager outputs | SF-08, SF-11 |
| External | The exported ONNX model | ONNX Runtime's outputs against PyTorch's | SF-07 |
| System | CLI, gate, traceability | Expected text; the gate's own failure cases | SF-13, SF-15 |

## 3. Pass/fail criteria

* Every test passes. A test verifying a requirement must pass for the requirement to count as verified.
* Arithmetic is compared **exactly** (integers). Time and memory are compared against stated limits.
* `ci/trace_matrix.py` reports no requirement without a passing test.

## 4. Entry and exit criteria

* **Entry:** the change builds, and ruff is clean.
* **Exit, for a merge:** all tests pass on Python 3.10 and 3.12 (GitHub Actions); the gate passes; the traceability matrix has no gaps.
* **Exit, for a release:** as for a merge; plus `results.md` regenerated, and a nightly sweep run with no unexplained change.

## 5. Environment

CPU-only PyTorch (≥ 2.6), transformers (≥ 4.50), onnx, onnxruntime; no GPU and no model
weights. Reference configurations are bundled. Hypothesis uses its default database;
failing examples are replayed first.

## 6. Deliverables

| Artefact | Produced by |
|---|---|
| JUnit XML, Cobertura coverage | `pytest` in Jenkins |
| `perf_report.md` | `ci/perf_gate.py` |
| `docs/traceability.md` | `ci/trace_matrix.py` |
| `examples/results.md` (the performance and accuracy report) | `examples/results.py` |
| `sweep.csv` | `ci/sweep.py` (nightly) |

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A PyTorch or transformers upgrade changes the decomposition | The gate separates arithmetic (fails) from operator-count drift (reported) |
| transformers takes a different code path under tracing | Characterised by a test; documented in the README |
| Timing tests are flaky on a loaded machine | Generous absolute limit (5 s against about 0.6 s); best of three in the gate |
| A new operator appears without a rule and is costed at zero | Coverage report names it; reference traces must have none |
