# Memory-Centric NPU DSE

A transparent, dependency-free starting framework for comparing **SRAM, eDRAM, eMRAM, and RRAM** in an Edge-AI NPU memory hierarchy.

It is deliberately an analytical first-order model—not a replacement for CACTI, SPICE, DRAMSim, or RTL. Its purpose is to make the architectural assumptions explicit early: capacity, banking, bandwidth, refresh, tiling/reloads, off-chip traffic, and end-to-end latency.

## What it models

- Conv-like and Transformer prefill/decode workload phases
- A banked NPU local memory with configurable capacity and bandwidth
- Weight residency / tiling and activation traffic
- Compute time versus local-memory and DRAM transfer time
- Dynamic local-memory energy, off-chip energy, and eDRAM refresh energy
- First-order area, soft-error, and write-endurance proxies
- Pareto filtering across latency, energy, and area

## Quick start

```bash
cd memory_npu_dse
python3 run_dse.py --config configs/edge_genai_baseline.json --out results
python3 -m unittest discover -s tests -v
```

The run writes `results/results.csv`, `results/pareto.csv`, and `results/summary.md`.

## Interpreting the baseline

The provided numbers are *illustrative assumptions*, chosen only to exercise the model. Replace them with technology-characterization data, CACTI/DESTINY/NVSim results, PDK data, or measured RTL/FPGA estimates before making research claims.

In particular, an eDRAM option is only credible when refresh cadence/energy and retention guardband are characterized for a selected technology and temperature range. NVM results should similarly include the actual read/write asymmetry, endurance, and reliability assumptions.

The RRAM point models **memory storage only**, not analog in-memory compute. Its `wear_limited_inferences` value assumes ideal wear levelling and is a screening proxy, not a lifetime claim. This makes it useful for exposing the key placement question: keep static weights in RRAM, but place frequently written activations and KV-cache state in SRAM/eDRAM unless endurance characterization supports another choice.

## Suggested research progression

1. Replace the synthetic workload table with ONNX-derived layer traces.
2. Add dataflow-specific tile equations for convolution/GEMM and a KV-cache allocator for decode.
3. Calibrate SRAM/eDRAM/NVM models using a consistent node and voltage target.
4. Add a banked scratchpad/refresh-controller RTL block and validate its transaction-level model.
5. Couple the DSE to SCALE-Sim/Timeloop or your existing mapping flow, then report full-model traffic and energy—not only array efficiency.

## Directory layout

```
configs/               Input assumptions and workload phases
memory_npu/            Reusable analytical model
run_dse.py              Command-line experiment runner
tests/                  Sanity checks for model invariants
```
