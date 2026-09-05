#!/usr/bin/env python3
"""Run the memory-centric NPU DSE from a JSON experiment description."""

import argparse
import csv
import json
from pathlib import Path

from memory_npu.model import evaluate, from_dicts


def pareto(points):
    """Keep points not dominated in latency, energy, and area for each workload."""
    kept = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and other["latency_ms"] <= candidate["latency_ms"]
            and other["total_energy_nj"] <= candidate["total_energy_nj"]
            and other["area_mm2"] <= candidate["area_mm2"]
            and (
                other["latency_ms"] < candidate["latency_ms"]
                or other["total_energy_nj"] < candidate["total_energy_nj"]
                or other["area_mm2"] < candidate["area_mm2"]
            )
            for other in points
        )
        if not dominated:
            kept.append(candidate)
    return kept


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    npu, memories, workloads = from_dicts(config)
    results = [evaluate(npu, memory, workload) for workload in workloads for memory in memories]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "results.csv", results)

    pareto_rows = []
    for workload in workloads:
        pareto_rows.extend(pareto([r for r in results if r["workload"] == workload.name]))
    write_csv(out / "pareto.csv", pareto_rows)

    lines = ["# Memory-Centric NPU DSE Summary", "", "| Workload | Memory | Tiles | DRAM MB | Latency ms | Energy µJ | Area mm² | Target |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r['workload']} | {r['memory']} | {r['weight_tiles']} | {r['dram_traffic_mb']:.2f} | "
            f"{r['latency_ms']:.3f} | {r['total_energy_nj']/1000:.3f} | {r['area_mm2']:.3f} | {'pass' if r['meets_target'] else 'miss'} |"
        )
    lines.extend(["", "Results are first-order estimates; see README for calibration requirements."])
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(results)} points to {out}")


if __name__ == "__main__":
    main()
