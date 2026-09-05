# Memory-Centric NPU DSE Summary

| Workload | Memory | Tiles | DRAM MB | Latency ms | Energy µJ | Area mm² | Target |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_cnn_block | SRAM-6T | 3 | 12.20 | 4.094 | 193.785 | 1.240 | pass |
| edge_cnn_block | eDRAM | 1 | 7.29 | 2.446 | 116.100 | 2.320 | pass |
| edge_cnn_block | eMRAM | 2 | 10.15 | 3.404 | 163.435 | 1.600 | pass |
| edge_cnn_block | RRAM | 2 | 10.15 | 3.404 | 173.748 | 0.880 | pass |
| transformer_prefill_128 | SRAM-6T | 10 | 71.77 | 24.084 | 1133.705 | 1.240 | pass |
| transformer_prefill_128 | eDRAM | 3 | 33.99 | 11.406 | 538.620 | 2.320 | pass |
| transformer_prefill_128 | eMRAM | 5 | 45.70 | 15.335 | 728.315 | 1.600 | pass |
| transformer_prefill_128 | RRAM | 5 | 45.70 | 15.335 | 751.118 | 0.880 | pass |
| transformer_decode_token | SRAM-6T | 10 | 69.06 | 23.172 | 1089.307 | 1.240 | miss |
| transformer_decode_token | eDRAM | 3 | 26.70 | 8.959 | 422.635 | 2.320 | pass |
| transformer_decode_token | eMRAM | 5 | 39.72 | 13.327 | 629.477 | 1.600 | miss |
| transformer_decode_token | RRAM | 5 | 39.72 | 13.327 | 631.079 | 0.880 | miss |

Results are first-order estimates; see README for calibration requirements.
