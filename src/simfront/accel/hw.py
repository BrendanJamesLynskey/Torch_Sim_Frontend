"""The modelled accelerator: one configuration object, and the time each part takes.

The block (see ``docs/accel_spec.md`` for its one-page specification)::

    off-chip memory ──(channels)──┐                 ┌── compute array (R x C MACs/cycle)
                                  ├─ NoC read  ─► on-chip buffer ─►┤
                                  └─ NoC write ◄─  (bytes)       ◄─┴── vector unit (lanes/cycle)
            load DMA engine moves tiles in; store DMA engine moves results out

Every number here is a parameter, and the presets are **illustrative**: they are
chosen to be the right order of magnitude for a small edge NPU and a datacentre-class
part, not taken from any product.

Times are in seconds. With ``quantise=True`` every duration is rounded up to whole
clock cycles and time is counted in cycles, which is what the cycle-stepped twin
(:mod:`simfront.accel.cycle`) needs to agree with the event-driven model exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

KiB, MiB, GB = 1024, 1024 * 1024, 1e9


@dataclass(frozen=True)
class AccelConfig:
    name: str = "edge-npu"
    clock_hz: float = 1.0e9
    array_rows: int = 32               # compute array: rows x cols multiply-accumulates per cycle
    array_cols: int = 32
    vector_lanes: int = 64             # vector unit: elements (or NTT butterflies) per cycle
    buffer_bytes: int = 2 * MiB        # on-chip buffer (SRAM)
    tile_fraction: float = 0.5         # a tile may use this share of the buffer (0.5 = double buffering)
    dram_bw: float = 25.6 * GB         # off-chip memory bandwidth, bytes/s, shared by its channels
    dram_channels: int = 2             # a transfer holds one channel at dram_bw / channels
    dram_latency: float = 100e-9       # per transfer
    noc_bw: float = 64 * GB            # interconnect, per direction (separate read and write networks)
    noc_latency: float = 20e-9
    transit_ops_per_byte: float | None = None   # speculative: a compute stage in the read path (None = off)
    transit_categories: tuple[str, ...] = ("ntt",)
    quantise: bool = False             # durations in whole cycles, time unit = cycles

    # ── derived rates ───────────────────────────────────────────────────
    @property
    def channel_bw(self) -> float:
        return self.dram_bw / self.dram_channels

    @property
    def path_bw(self) -> float:
        """A single transfer's bandwidth: the narrower of one memory channel and the NoC."""
        return min(self.channel_bw, self.noc_bw)

    @property
    def path_limit(self) -> str:
        return "dram" if self.channel_bw <= self.noc_bw else "noc"

    @property
    def tile_budget(self) -> int:
        return int(self.buffer_bytes * self.tile_fraction)

    @property
    def peak_macs(self) -> float:
        return self.array_rows * self.array_cols * self.clock_hz

    def with_(self, **kw) -> "AccelConfig":
        return replace(self, **kw)

    # ── durations ───────────────────────────────────────────────────────
    def _time(self, seconds: float) -> float:
        if not self.quantise:
            return seconds
        return float(math.ceil(seconds * self.clock_hz - 1e-9)) if seconds > 0 else 0.0

    def transfer_time(self, nbytes: int, ops: int = 0) -> float:
        """One DMA transfer: memory + NoC latency, then the bytes at the path bandwidth.

        With an in-transit stage, a transfer carrying ``ops`` operations is slowed to the
        stage's rate when it needs more than ``transit_ops_per_byte`` per byte.
        """
        if nbytes <= 0:
            return 0.0
        t = nbytes / self.path_bw
        if ops and self.transit_ops_per_byte:
            t = max(t, ops / (self.transit_ops_per_byte * self.path_bw))
        return self._time(self.dram_latency + self.noc_latency + t)

    def array_cycles(self, batch: int, m: int, k: int, n: int) -> int:
        """Cycle-approximate output-stationary systolic array.

        Each R x C block of outputs takes ``k`` cycles to accumulate; filling and draining the
        array adds R + C cycles once per tile (later blocks overlap the drain of earlier ones).
        """
        r, c = self.array_rows, self.array_cols
        return batch * math.ceil(m / r) * math.ceil(n / c) * k + r + c

    def array_time(self, batch: int, m: int, k: int, n: int) -> float:
        return self._time(self.array_cycles(batch, m, k, n) / self.clock_hz)

    def vector_time(self, work: int) -> float:
        return self._time(math.ceil(work / self.vector_lanes) / self.clock_hz) if work > 0 else 0.0


PRESETS = {
    # A small edge NPU: 1,024 MACs/cycle at 1 GHz (2 TFLOP/s), 2 MiB SRAM, LPDDR-class memory.
    "edge-npu": AccelConfig(),
    # A datacentre-class part: 128 x 128 array (32.8 TFLOP/s at 1 GHz), 32 MiB SRAM, HBM-class memory.
    "dc-npu": AccelConfig(name="dc-npu", array_rows=128, array_cols=128, vector_lanes=512, buffer_bytes=32 * MiB,
                          dram_bw=1600 * GB, dram_channels=8, noc_bw=512 * GB),
}


def preset(name: str, **kw) -> AccelConfig:
    return PRESETS[name].with_(**kw)
