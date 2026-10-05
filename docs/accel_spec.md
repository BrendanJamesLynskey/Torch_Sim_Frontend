# simfront.accel: block specification (one page)

Version 1.0. The performance model of a tiled matrix accelerator, written the way a hardware
block's specification is written: what it contains, what goes in and out, how it behaves, how
long things take, what it reports, and what it deliberately leaves out. The requirements it must
meet are SF-18 to SF-28 in [`spec.md`](spec.md); every number it produces is in
[`../examples/accel_results.md`](../examples/accel_results.md). All parameter values are
**illustrative**.

## 1. Block diagram and contents

```
                 ┌──────────────── accelerator ────────────────────────────────────┐
 off-chip  ◄────►│ memory channels (dram_channels)                                 │
 memory          │   │  NoC read (noc_bw)            NoC write (noc_bw)  ▲          │
                 │   ▼                                                   │          │
                 │ load DMA ──► on-chip buffer (buffer_bytes) ──► store DMA          │
                 │                 │  ready FIFO        ▲ done FIFO                  │
                 │                 ▼                    │                           │
                 │       compute: array (rows x cols MAC/cycle) │ vector unit (lanes) │
                 │       [optional, speculative: in-transit stage in the read path]  │
                 └──────────────────────────────────────────────────────────────────┘
```

| Sub-block | SimPy primitive | Parameters (`AccelConfig`) |
|---|---|---|
| Off-chip memory | `Resource(dram_channels)` | `dram_bw` (shared by the channels), `dram_latency` |
| Interconnect (NoC) | one `Resource(1)` per direction | `noc_bw` per direction, `noc_latency` |
| Load / store DMA | one process each | (none) |
| On-chip buffer | `Container(buffer_bytes)` | `buffer_bytes`, `tile_fraction` |
| Ready / done FIFOs | `Store()` | unbounded: the buffer bounds them |
| Compute array | `Resource(1)` | `array_rows`, `array_cols`, `clock_hz` |
| Vector unit | `Resource(1)` | `vector_lanes` |
| In-transit stage | part of a load | `transit_ops_per_byte`, `transit_categories` |

Presets: `edge-npu` (32 x 32 array, 2 MiB, 25.6 GB/s) and `dc-npu` (128 x 128, 32 MiB, 1.6 TB/s).

## 2. Interfaces

* **Input:** a `simfront-trace/1` operator trace from any front end (torch.export, ONNX, dispatch,
  torch.compile) or a workload builder (`ntt.polymul_trace`), plus an `AccelConfig`.
* **Output:** per-tile timings (`Timings`) and a `Report`: latency, utilisation per component,
  stall breakdown, hot-spot, per-operator table and histogram; a timeline PNG and a Chrome trace.

## 3. Behaviour

1. **Lowering.** Each operator becomes tiles (see `lower.py`): GEMMs (matmul, convolution as
   im2col, attention) are blocked so that operands and result fit in `tile_fraction x buffer_bytes`;
   other operators are cut into equal byte slices. Views are free; operators without a cost rule are
   reported and not simulated.
2. **Load (in program order).** For the first tile of an operator, wait until every producer of
   its activations has been stored. Allocate the tile's operand and result bytes in the buffer,
   **waiting while the buffer is full (back-pressure)**. Hold one memory channel and the NoC read
   network for the transfer. Put the tile in the ready FIFO.
3. **Compute (in order, single issue).** Take the next ready tile; run it on its unit; free its
   operand bytes. A tile that is not the last K-slice of its block frees its partial sums too and
   stores nothing.
4. **Store (in order).** Hold one memory channel and the NoC write network; free the result bytes;
   the operator's last tile marks the operator done.

## 4. Timing model

| Event | Duration |
|---|---|
| Transfer of *b* bytes | `dram_latency + noc_latency + b / min(dram_bw / dram_channels, noc_bw)` |
| ... in transit, carrying *o* operations | the same, with `b / bw` replaced by `max(b / bw, o / (transit_ops_per_byte x bw))` |
| Array tile (batch, *m*, *k*, *n*) | `batch x ceil(m/rows) x ceil(n/cols) x k + rows + cols` cycles (output-stationary; fill and drain once per tile) |
| Vector tile of *w* elements (or NTT butterflies) | `ceil(w / lanes)` cycles |

With `quantise=True` every duration is rounded up to whole cycles and time is counted in cycles.

## 5. Ordering, hazards and determinism

* Read-after-write through memory: an operator's loads start only after its producers' stores end.
* No write-after-read hazard arises: every result is written to a new tensor.
* With `dram_channels >= 2` the two DMA engines never contend; with one channel they arbitrate
  first come, first served, and only the SimPy engine models it (the fast path refuses).
* Same program, same timings: there is no randomness in the block.

## 6. Performance counters (outputs)

Busy fraction of every component; bytes and bandwidth used on memory and the NoC; PE efficiency
of the array (useful MACs over peak MACs while busy); buffer mean and peak occupancy; ready-FIFO
depth over time; compute time split into computing, load bandwidth, buffer full, dependency and
store tail (summing to the latency); the hot-spot; per-operator latency and bound.

## 7. Equivalent engines

The same `Program` runs on three engines that must agree: the SimPy model (reference), the
recurrence fast path (Python, and C++20 via pybind11: bit-identical when the DMAs never contend),
and the cycle-stepped twin (identical cycle for cycle on a quantised program).

## 8. Not modelled (assumptions)

Operand reuse across tiles (each tile loads both operand blocks); on-chip fusion between operators
(every result round-trips through off-chip memory); out-of-order or concurrent issue on the array
and vector unit; DRAM banks, rows and refresh (see Memory_System_Sim for those); the NoC as a
network (it is two links); power and area; the accuracy of any preset against real hardware.
