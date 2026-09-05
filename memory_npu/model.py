"""Transparent first-order PPA and traffic model for a memory-centric NPU."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Optional


BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class MemoryTechnology:
    name: str
    kind: str
    capacity_kb: int
    bandwidth_gbps: float
    read_energy_pj_per_byte: float
    write_energy_pj_per_byte: float
    area_mm2_per_mb: float
    retention_ms: Optional[float]
    refresh_interval_us: Optional[float]
    refresh_energy_nj_per_mb: float
    soft_error_fit_per_mb: float
    endurance_cycles: Optional[int]

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_kb * 1024

    @property
    def area_mm2(self) -> float:
        return self.capacity_bytes / BYTES_PER_MB * self.area_mm2_per_mb


@dataclass(frozen=True)
class NPUConfig:
    peak_macs_per_cycle: int
    frequency_mhz: float
    dram_bandwidth_gbps: float
    dram_energy_pj_per_byte: float
    utilization: float
    banks: int

    @property
    def effective_macs_per_s(self) -> float:
        return self.peak_macs_per_cycle * self.frequency_mhz * 1e6 * self.utilization


@dataclass(frozen=True)
class Workload:
    name: str
    phase: str
    macs: int
    weight_bytes: int
    activation_read_bytes: int
    activation_write_bytes: int
    reuse_factor: float
    target_latency_ms: float
    kv_cache_read_bytes: int = 0
    kv_cache_write_bytes: int = 0

    @property
    def activation_bytes(self) -> int:
        return self.activation_read_bytes + self.activation_write_bytes

    @property
    def kv_cache_bytes(self) -> int:
        return self.kv_cache_read_bytes + self.kv_cache_write_bytes


def _seconds_for_transfer(byte_count: float, bandwidth_gbps: float) -> float:
    if bandwidth_gbps <= 0:
        raise ValueError("Bandwidth must be positive")
    return byte_count * 8 / (bandwidth_gbps * 1e9)


def evaluate(npu: NPUConfig, memory: MemoryTechnology, workload: Workload) -> dict:
    """Estimate one workload-memory point.

    Assumption: the local store is shared between weight tiles and a tiled live
    activation/KV working set. At most 40% of the store is reserved for live
    state; the remaining space holds a weight tile. When the whole state does
    not fit, it is tiled and the nonresident portion contributes to spill traffic.
    """
    if not 0 < npu.utilization <= 1:
        raise ValueError("NPU utilization must be in (0, 1]")
    if workload.reuse_factor <= 0:
        raise ValueError("Reuse factor must be positive")

    live_state = workload.activation_bytes + workload.kv_cache_bytes
    resident_state = min(live_state, int(memory.capacity_bytes * 0.40))
    usable_for_weights = max(1, memory.capacity_bytes - resident_state)
    weight_tiles = ceil(workload.weight_bytes / usable_for_weights)
    weights_fit = workload.weight_bytes <= usable_for_weights

    # A spill factor is intentionally conservative: extra tiles induce a partial
    # activation/KV reload in addition to the weight reload itself.
    spill_rounds = max(0, weight_tiles - 1)
    nonresident_state = max(0, live_state - resident_state)
    spilled_state_bytes = nonresident_state + spill_rounds * (live_state / workload.reuse_factor)
    dram_bytes = workload.weight_bytes + spilled_state_bytes
    local_read_bytes = workload.weight_bytes + workload.activation_read_bytes + workload.kv_cache_read_bytes
    local_write_bytes = workload.activation_write_bytes + workload.kv_cache_write_bytes

    compute_s = workload.macs / npu.effective_macs_per_s
    local_s = _seconds_for_transfer(local_read_bytes + local_write_bytes, memory.bandwidth_gbps)
    dram_s = _seconds_for_transfer(dram_bytes, npu.dram_bandwidth_gbps)
    # Local and DRAM transfers may overlap with compute; DRAM is the exposed stream.
    latency_s = max(compute_s, local_s, dram_s)

    local_energy_nj = (
        local_read_bytes * memory.read_energy_pj_per_byte
        + local_write_bytes * memory.write_energy_pj_per_byte
    ) / 1e3
    dram_energy_nj = dram_bytes * npu.dram_energy_pj_per_byte / 1e3
    refresh_count = 0 if not memory.refresh_interval_us else ceil(latency_s * 1e6 / memory.refresh_interval_us)
    refresh_energy_nj = refresh_count * (memory.capacity_bytes / BYTES_PER_MB) * memory.refresh_energy_nj_per_mb
    total_energy_nj = local_energy_nj + dram_energy_nj + refresh_energy_nj
    latency_ms = latency_s * 1e3
    equivalent_array_writes = local_write_bytes / memory.capacity_bytes
    endurance_inferences = (
        memory.endurance_cycles / equivalent_array_writes
        if memory.endurance_cycles and equivalent_array_writes > 0
        else None
    )

    return {
        "workload": workload.name,
        "phase": workload.phase,
        "memory": memory.name,
        "memory_kind": memory.kind,
        "capacity_kb": memory.capacity_kb,
        "banks": npu.banks,
        "weight_tiles": weight_tiles,
        "weights_fit": weights_fit,
        "resident_state_mb": resident_state / BYTES_PER_MB,
        "dram_traffic_mb": dram_bytes / BYTES_PER_MB,
        "spill_traffic_mb": spilled_state_bytes / BYTES_PER_MB,
        "local_traffic_mb": (local_read_bytes + local_write_bytes) / BYTES_PER_MB,
        "compute_latency_ms": compute_s * 1e3,
        "local_transfer_ms": local_s * 1e3,
        "dram_transfer_ms": dram_s * 1e3,
        "latency_ms": latency_ms,
        "target_latency_ms": workload.target_latency_ms,
        "meets_target": latency_ms <= workload.target_latency_ms,
        "local_energy_nj": local_energy_nj,
        "dram_energy_nj": dram_energy_nj,
        "refresh_energy_nj": refresh_energy_nj,
        "endurance_cycles": memory.endurance_cycles,
        "array_equivalent_writes_per_inference": equivalent_array_writes,
        "wear_limited_inferences": endurance_inferences,
        "total_energy_nj": total_energy_nj,
        "area_mm2": memory.area_mm2,
        "soft_error_fit": memory.soft_error_fit_per_mb * memory.capacity_bytes / BYTES_PER_MB,
    }


def from_dicts(config: dict) -> tuple[NPUConfig, list[MemoryTechnology], list[Workload]]:
    return (
        NPUConfig(**config["npu"]),
        [MemoryTechnology(**item) for item in config["memories"]],
        [Workload(**item) for item in config["workloads"]],
    )
