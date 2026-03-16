# MP-SENet Lite Shrink Experiment

| Variant | Params | FP32 weights | Int8 weights | Peak activation | Workspace | MAC/s | Latency | CPU mean | CUDA mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current 16 kHz | 311652 | 1.19 MiB | 0.30 MiB | 7.94 MiB | 15.87 MiB | 7.27 G | 1000 ms | 156.99 ms | 7.38 ms |
| tiny 8 kHz | 24900 | 0.09 MiB | 0.02 MiB | 0.50 MiB | 1.51 MiB | 0.15 G | 1008 ms | 15.42 ms | 2.53 ms |

Reductions for tiny 8 kHz vs current 16 kHz:

- params/int8 weights: 0.080x
- peak activation: 0.063x
- workspace: 0.095x
- MAC/s: 0.020x