# Embedded Real-Time Under-50mW Comparison

This report centralizes all currently explored deployment ideas under the product constraint:

- real-time
- single independent chip
- preferably under 50 mW

The data comes from `reports/mcu_feasibility_stretch.json` and includes both on-chip and external-memory single-chip options.
Power values are simulator estimates, not oscilloscope or board measurements.

## Primary Conclusions

- Quality-first single-chip option: `percepnet_class_16k` on `Alif Ensemble E3` at `16 kHz`, `1.26 mW`, `onchip`.
- Efficiency-first single-chip option: `rnnoise_class_8k` on `Alif Ensemble E3` at `8 kHz`, `0.33 mW`, `onchip`.
- `STM32L476RG` remains suitable only for very small paths such as `spectral_gate_only` and `rnnoise_class_8k`.
- `MP-SENet-lite` and `aTENNuate` current repo variants are not viable for this budget because they remain offline and/or too compute-heavy.
- `MP-SENet-micro` is the first redesign target that becomes realistic on `NXP MCX N94` and `Alif Ensemble E3`.

## Best On-Chip Candidates

| Model | Hardware | Bandwidth | Quality | Memory | Flash | SRAM | Latency | RTF | Power | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | Alif Ensemble E3 | 16 kHz | high | onchip | 0.31 MiB | 0.20 MiB | 20 ms | 0.03 | 1.26 mW | PASS |
| percepnet_class_16k | NXP MCX N94 | 16 kHz | high | onchip | 0.31 MiB | 0.20 MiB | 20 ms | 0.52 | 14.44 mW | PASS |
| rnnoise_class_16k | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.16 MiB | 0.07 MiB | 20 ms | 0.02 | 0.67 mW | PASS |
| spectral_gate_only_16k | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.02 | 0.81 mW | PASS |
| mp_senet_micro_16k_target | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.35 MiB | 0.26 MiB | 20 ms | 0.03 | 1.00 mW | PASS |
| spectral_gate_only_16k | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.14 | 3.88 mW | PASS |
| mp_senet_micro_16k_target | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.35 MiB | 0.26 MiB | 20 ms | 0.17 | 4.82 mW | PASS |
| rnnoise_class_16k | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.16 MiB | 0.07 MiB | 20 ms | 0.26 | 7.18 mW | PASS |

## All Single-Chip PASS Candidates Under 50 mW

| Model | Hardware | Bandwidth | Quality | Memory | Flash | SRAM | Latency | RTF | Power | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | Alif Ensemble E3 | 16 kHz | high | onchip | 0.31 MiB | 0.20 MiB | 20 ms | 0.03 | 1.26 mW | PASS |
| percepnet_class_16k | NXP MCX N94 | 16 kHz | high | onchip | 0.31 MiB | 0.20 MiB | 20 ms | 0.52 | 14.44 mW | PASS |
| rnnoise_class_16k | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.16 MiB | 0.07 MiB | 20 ms | 0.02 | 0.67 mW | PASS |
| spectral_gate_only_16k | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.02 | 0.81 mW | PASS |
| mp_senet_micro_16k_target | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.35 MiB | 0.26 MiB | 20 ms | 0.03 | 1.00 mW | PASS |
| spectral_gate_only_16k | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.14 | 3.88 mW | PASS |
| mp_senet_micro_16k_target | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.35 MiB | 0.26 MiB | 20 ms | 0.17 | 4.82 mW | PASS |
| rnnoise_class_16k | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.16 MiB | 0.07 MiB | 20 ms | 0.26 | 7.18 mW | PASS |
| spectral_gate_only_16k | STM32L476RG | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.61 | 16.13 mW | PASS |
| mp_senet_micro_8k_target | Alif Ensemble E3 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.18 MiB | 20 ms | 0.01 | 0.48 mW | PASS |
| percepnet_class_8k | Alif Ensemble E3 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.10 MiB | 20 ms | 0.02 | 0.63 mW | PASS |
| mp_senet_micro_8k_target | NXP MCX N94 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.18 MiB | 20 ms | 0.10 | 2.66 mW | PASS |
| percepnet_class_8k | NXP MCX N94 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.10 MiB | 20 ms | 0.26 | 7.22 mW | PASS |
| rnnoise_class_8k | Alif Ensemble E3 | 8 kHz | degraded | onchip | 0.14 MiB | 0.04 MiB | 20 ms | 0.01 | 0.33 mW | PASS |
| spectral_gate_only_8k | Alif Ensemble E3 | 8 kHz | degraded | onchip | 0.05 MiB | 0.03 MiB | 32 ms | 0.01 | 0.41 mW | PASS |
| spectral_gate_only_8k | NXP MCX N94 | 8 kHz | degraded | onchip | 0.05 MiB | 0.03 MiB | 32 ms | 0.07 | 1.94 mW | PASS |
| rnnoise_class_8k | NXP MCX N94 | 8 kHz | degraded | onchip | 0.14 MiB | 0.04 MiB | 20 ms | 0.13 | 3.59 mW | PASS |
| spectral_gate_only_8k | STM32L476RG | 8 kHz | degraded | onchip | 0.05 MiB | 0.03 MiB | 32 ms | 0.31 | 8.07 mW | PASS |
| rnnoise_class_8k | STM32L476RG | 8 kHz | degraded | onchip | 0.14 MiB | 0.04 MiB | 20 ms | 0.99 | 26.03 mW | PASS |
| percepnet_class_16k | Infineon PSoC Edge E84 | 16 kHz | high | external | 0.31 MiB | 0.20 MiB | 20 ms | 0.05 | 10.20 mW | PASS |
| percepnet_class_16k | STM32N6 | 16 kHz | high | external | 0.31 MiB | 0.20 MiB | 20 ms | 0.02 | 13.63 mW | PASS |
| percepnet_class_16k | NXP i.MX RT700 | 16 kHz | high | external | 0.31 MiB | 0.20 MiB | 20 ms | 0.21 | 17.27 mW | PASS |
| rnnoise_class_16k | Infineon PSoC Edge E84 | 16 kHz | acceptable | external | 0.16 MiB | 0.07 MiB | 20 ms | 0.02 | 9.11 mW | PASS |
| spectral_gate_only_16k | Infineon PSoC Edge E84 | 16 kHz | acceptable | external | 0.05 MiB | 0.05 MiB | 32 ms | 0.03 | 9.20 mW | PASS |
| mp_senet_micro_16k_target | Infineon PSoC Edge E84 | 16 kHz | acceptable | external | 0.35 MiB | 0.26 MiB | 20 ms | 0.04 | 9.67 mW | PASS |
| spectral_gate_only_16k | NXP i.MX RT700 | 16 kHz | acceptable | external | 0.05 MiB | 0.05 MiB | 32 ms | 0.05 | 10.42 mW | PASS |
| mp_senet_micro_16k_target | NXP i.MX RT700 | 16 kHz | acceptable | external | 0.35 MiB | 0.26 MiB | 20 ms | 0.06 | 10.79 mW | PASS |
| rnnoise_class_16k | STM32N6 | 16 kHz | acceptable | external | 0.16 MiB | 0.07 MiB | 20 ms | 0.01 | 11.93 mW | PASS |
| spectral_gate_only_16k | STM32N6 | 16 kHz | acceptable | external | 0.05 MiB | 0.05 MiB | 32 ms | 0.01 | 12.40 mW | PASS |
| rnnoise_class_16k | NXP i.MX RT700 | 16 kHz | acceptable | external | 0.16 MiB | 0.07 MiB | 20 ms | 0.10 | 12.64 mW | PASS |
| mp_senet_micro_16k_target | STM32N6 | 16 kHz | acceptable | external | 0.35 MiB | 0.26 MiB | 20 ms | 0.02 | 12.90 mW | PASS |
| mp_senet_micro_8k_target | Infineon PSoC Edge E84 | 8 kHz | acceptable | external | 0.27 MiB | 0.18 MiB | 20 ms | 0.02 | 8.81 mW | PASS |
| percepnet_class_8k | Infineon PSoC Edge E84 | 8 kHz | acceptable | external | 0.27 MiB | 0.10 MiB | 20 ms | 0.02 | 9.10 mW | PASS |
| mp_senet_micro_8k_target | NXP i.MX RT700 | 8 kHz | acceptable | external | 0.27 MiB | 0.18 MiB | 20 ms | 0.04 | 9.57 mW | PASS |
| mp_senet_micro_8k_target | STM32N6 | 8 kHz | acceptable | external | 0.27 MiB | 0.18 MiB | 20 ms | 0.01 | 11.39 mW | PASS |
| percepnet_class_8k | STM32N6 | 8 kHz | acceptable | external | 0.27 MiB | 0.10 MiB | 20 ms | 0.01 | 11.81 mW | PASS |
| percepnet_class_8k | NXP i.MX RT700 | 8 kHz | acceptable | external | 0.27 MiB | 0.10 MiB | 20 ms | 0.11 | 12.64 mW | PASS |
| rnnoise_class_8k | Infineon PSoC Edge E84 | 8 kHz | degraded | external | 0.14 MiB | 0.04 MiB | 20 ms | 0.01 | 8.56 mW | PASS |
| spectral_gate_only_8k | Infineon PSoC Edge E84 | 8 kHz | degraded | external | 0.05 MiB | 0.03 MiB | 32 ms | 0.01 | 8.60 mW | PASS |
| spectral_gate_only_8k | NXP i.MX RT700 | 8 kHz | degraded | external | 0.05 MiB | 0.03 MiB | 32 ms | 0.03 | 9.21 mW | PASS |
| rnnoise_class_8k | NXP i.MX RT700 | 8 kHz | degraded | external | 0.14 MiB | 0.04 MiB | 20 ms | 0.05 | 10.32 mW | PASS |
| rnnoise_class_8k | STM32N6 | 8 kHz | degraded | external | 0.14 MiB | 0.04 MiB | 20 ms | 0.01 | 10.97 mW | PASS |
| spectral_gate_only_8k | STM32N6 | 8 kHz | degraded | external | 0.05 MiB | 0.03 MiB | 32 ms | 0.01 | 11.20 mW | PASS |

## On-Chip Only PASS Candidates Under 50 mW

| Model | Hardware | Bandwidth | Quality | Memory | Flash | SRAM | Latency | RTF | Power | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | Alif Ensemble E3 | 16 kHz | high | onchip | 0.31 MiB | 0.20 MiB | 20 ms | 0.03 | 1.26 mW | PASS |
| percepnet_class_16k | NXP MCX N94 | 16 kHz | high | onchip | 0.31 MiB | 0.20 MiB | 20 ms | 0.52 | 14.44 mW | PASS |
| rnnoise_class_16k | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.16 MiB | 0.07 MiB | 20 ms | 0.02 | 0.67 mW | PASS |
| spectral_gate_only_16k | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.02 | 0.81 mW | PASS |
| mp_senet_micro_16k_target | Alif Ensemble E3 | 16 kHz | acceptable | onchip | 0.35 MiB | 0.26 MiB | 20 ms | 0.03 | 1.00 mW | PASS |
| spectral_gate_only_16k | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.14 | 3.88 mW | PASS |
| mp_senet_micro_16k_target | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.35 MiB | 0.26 MiB | 20 ms | 0.17 | 4.82 mW | PASS |
| rnnoise_class_16k | NXP MCX N94 | 16 kHz | acceptable | onchip | 0.16 MiB | 0.07 MiB | 20 ms | 0.26 | 7.18 mW | PASS |
| spectral_gate_only_16k | STM32L476RG | 16 kHz | acceptable | onchip | 0.05 MiB | 0.05 MiB | 32 ms | 0.61 | 16.13 mW | PASS |
| mp_senet_micro_8k_target | Alif Ensemble E3 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.18 MiB | 20 ms | 0.01 | 0.48 mW | PASS |
| percepnet_class_8k | Alif Ensemble E3 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.10 MiB | 20 ms | 0.02 | 0.63 mW | PASS |
| mp_senet_micro_8k_target | NXP MCX N94 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.18 MiB | 20 ms | 0.10 | 2.66 mW | PASS |
| percepnet_class_8k | NXP MCX N94 | 8 kHz | acceptable | onchip | 0.27 MiB | 0.10 MiB | 20 ms | 0.26 | 7.22 mW | PASS |
| rnnoise_class_8k | Alif Ensemble E3 | 8 kHz | degraded | onchip | 0.14 MiB | 0.04 MiB | 20 ms | 0.01 | 0.33 mW | PASS |
| spectral_gate_only_8k | Alif Ensemble E3 | 8 kHz | degraded | onchip | 0.05 MiB | 0.03 MiB | 32 ms | 0.01 | 0.41 mW | PASS |
| spectral_gate_only_8k | NXP MCX N94 | 8 kHz | degraded | onchip | 0.05 MiB | 0.03 MiB | 32 ms | 0.07 | 1.94 mW | PASS |
| rnnoise_class_8k | NXP MCX N94 | 8 kHz | degraded | onchip | 0.14 MiB | 0.04 MiB | 20 ms | 0.13 | 3.59 mW | PASS |
| spectral_gate_only_8k | STM32L476RG | 8 kHz | degraded | onchip | 0.05 MiB | 0.03 MiB | 32 ms | 0.31 | 8.07 mW | PASS |
| rnnoise_class_8k | STM32L476RG | 8 kHz | degraded | onchip | 0.14 MiB | 0.04 MiB | 20 ms | 0.99 | 26.03 mW | PASS |

## External-Memory Single-Chip PASS Candidates Under 50 mW

| Model | Hardware | Bandwidth | Quality | Memory | Flash | SRAM | Latency | RTF | Power | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | Infineon PSoC Edge E84 | 16 kHz | high | external | 0.31 MiB | 0.20 MiB | 20 ms | 0.05 | 10.20 mW | PASS |
| percepnet_class_16k | STM32N6 | 16 kHz | high | external | 0.31 MiB | 0.20 MiB | 20 ms | 0.02 | 13.63 mW | PASS |
| percepnet_class_16k | NXP i.MX RT700 | 16 kHz | high | external | 0.31 MiB | 0.20 MiB | 20 ms | 0.21 | 17.27 mW | PASS |
| rnnoise_class_16k | Infineon PSoC Edge E84 | 16 kHz | acceptable | external | 0.16 MiB | 0.07 MiB | 20 ms | 0.02 | 9.11 mW | PASS |
| spectral_gate_only_16k | Infineon PSoC Edge E84 | 16 kHz | acceptable | external | 0.05 MiB | 0.05 MiB | 32 ms | 0.03 | 9.20 mW | PASS |
| mp_senet_micro_16k_target | Infineon PSoC Edge E84 | 16 kHz | acceptable | external | 0.35 MiB | 0.26 MiB | 20 ms | 0.04 | 9.67 mW | PASS |
| spectral_gate_only_16k | NXP i.MX RT700 | 16 kHz | acceptable | external | 0.05 MiB | 0.05 MiB | 32 ms | 0.05 | 10.42 mW | PASS |
| mp_senet_micro_16k_target | NXP i.MX RT700 | 16 kHz | acceptable | external | 0.35 MiB | 0.26 MiB | 20 ms | 0.06 | 10.79 mW | PASS |
| rnnoise_class_16k | STM32N6 | 16 kHz | acceptable | external | 0.16 MiB | 0.07 MiB | 20 ms | 0.01 | 11.93 mW | PASS |
| spectral_gate_only_16k | STM32N6 | 16 kHz | acceptable | external | 0.05 MiB | 0.05 MiB | 32 ms | 0.01 | 12.40 mW | PASS |
| rnnoise_class_16k | NXP i.MX RT700 | 16 kHz | acceptable | external | 0.16 MiB | 0.07 MiB | 20 ms | 0.10 | 12.64 mW | PASS |
| mp_senet_micro_16k_target | STM32N6 | 16 kHz | acceptable | external | 0.35 MiB | 0.26 MiB | 20 ms | 0.02 | 12.90 mW | PASS |
| mp_senet_micro_8k_target | Infineon PSoC Edge E84 | 8 kHz | acceptable | external | 0.27 MiB | 0.18 MiB | 20 ms | 0.02 | 8.81 mW | PASS |
| percepnet_class_8k | Infineon PSoC Edge E84 | 8 kHz | acceptable | external | 0.27 MiB | 0.10 MiB | 20 ms | 0.02 | 9.10 mW | PASS |
| mp_senet_micro_8k_target | NXP i.MX RT700 | 8 kHz | acceptable | external | 0.27 MiB | 0.18 MiB | 20 ms | 0.04 | 9.57 mW | PASS |
| mp_senet_micro_8k_target | STM32N6 | 8 kHz | acceptable | external | 0.27 MiB | 0.18 MiB | 20 ms | 0.01 | 11.39 mW | PASS |
| percepnet_class_8k | STM32N6 | 8 kHz | acceptable | external | 0.27 MiB | 0.10 MiB | 20 ms | 0.01 | 11.81 mW | PASS |
| percepnet_class_8k | NXP i.MX RT700 | 8 kHz | acceptable | external | 0.27 MiB | 0.10 MiB | 20 ms | 0.11 | 12.64 mW | PASS |
| rnnoise_class_8k | Infineon PSoC Edge E84 | 8 kHz | degraded | external | 0.14 MiB | 0.04 MiB | 20 ms | 0.01 | 8.56 mW | PASS |
| spectral_gate_only_8k | Infineon PSoC Edge E84 | 8 kHz | degraded | external | 0.05 MiB | 0.03 MiB | 32 ms | 0.01 | 8.60 mW | PASS |
| spectral_gate_only_8k | NXP i.MX RT700 | 8 kHz | degraded | external | 0.05 MiB | 0.03 MiB | 32 ms | 0.03 | 9.21 mW | PASS |
| rnnoise_class_8k | NXP i.MX RT700 | 8 kHz | degraded | external | 0.14 MiB | 0.04 MiB | 20 ms | 0.05 | 10.32 mW | PASS |
| rnnoise_class_8k | STM32N6 | 8 kHz | degraded | external | 0.14 MiB | 0.04 MiB | 20 ms | 0.01 | 10.97 mW | PASS |
| spectral_gate_only_8k | STM32N6 | 8 kHz | degraded | external | 0.05 MiB | 0.03 MiB | 32 ms | 0.01 | 11.20 mW | PASS |

## Current Repo Models: Why They Still Fail

| Model | Hardware | Bandwidth | Memory | Flash | SRAM | Latency | RTF | Verdict | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atennuate_8k_estimated | Alif Ensemble E3 | 8 kHz | onchip | 0.90 MiB | 0.16 MiB | 1024 ms | 0.23 | FAIL | latency,offline_only |
| atennuate_8k_estimated | Infineon PSoC Edge E84 | 8 kHz | external | 0.90 MiB | 0.16 MiB | 1024 ms | 0.26 | FAIL | latency,offline_only |
| atennuate_8k_estimated | NXP MCX N94 | 8 kHz | onchip | 0.90 MiB | 0.16 MiB | 1024 ms | 1.12 | FAIL | compute,latency,offline_only |
| atennuate_8k_estimated | NXP i.MX RT700 | 8 kHz | external | 0.90 MiB | 0.16 MiB | 1024 ms | 0.42 | FAIL | latency,offline_only |
| atennuate_8k_estimated | STM32L476RG | 8 kHz | onchip | 0.90 MiB | 0.16 MiB | 1024 ms | 7.29 | FAIL | compute,flash,latency,offline_only,sram |
| atennuate_8k_estimated | STM32N6 | 8 kHz | external | 0.90 MiB | 0.16 MiB | 1024 ms | 0.13 | FAIL | latency,offline_only |
| atennuate_16k | Alif Ensemble E3 | 16 kHz | onchip | 0.90 MiB | 0.31 MiB | 1024 ms | 0.46 | FAIL | latency,offline_only |
| atennuate_16k | Infineon PSoC Edge E84 | 16 kHz | external | 0.90 MiB | 0.31 MiB | 1024 ms | 0.51 | FAIL | latency,offline_only |
| atennuate_16k | NXP MCX N94 | 16 kHz | onchip | 0.90 MiB | 0.31 MiB | 1024 ms | 2.25 | FAIL | compute,latency,offline_only |
| atennuate_16k | NXP i.MX RT700 | 16 kHz | external | 0.90 MiB | 0.31 MiB | 1024 ms | 0.85 | FAIL | latency,offline_only |
| atennuate_16k | STM32L476RG | 16 kHz | onchip | 0.90 MiB | 0.31 MiB | 1024 ms | 14.39 | FAIL | compute,flash,latency,offline_only,sram |
| atennuate_16k | STM32N6 | 16 kHz | external | 0.90 MiB | 0.31 MiB | 1024 ms | 0.26 | FAIL | latency,offline_only |
| mp_senet_lite_8k_estimated | Alif Ensemble E3 | 8 kHz | onchip | 0.42 MiB | 1.22 MiB | 1000 ms | 2.74 | FAIL | compute,latency,offline_only |
| mp_senet_lite_8k_estimated | Infineon PSoC Edge E84 | 8 kHz | external | 0.42 MiB | 1.22 MiB | 1000 ms | 3.24 | FAIL | compute,latency,offline_only |
| mp_senet_lite_8k_estimated | NXP MCX N94 | 8 kHz | onchip | 0.42 MiB | 1.22 MiB | 1000 ms | 42.72 | FAIL | compute,latency,offline_only,sram |
| mp_senet_lite_8k_estimated | NXP i.MX RT700 | 8 kHz | external | 0.42 MiB | 1.22 MiB | 1000 ms | 17.92 | FAIL | compute,latency,offline_only |
| mp_senet_lite_8k_estimated | STM32L476RG | 8 kHz | onchip | 0.42 MiB | 1.22 MiB | 1000 ms | 169.44 | FAIL | compute,latency,offline_only,sram |
| mp_senet_lite_8k_estimated | STM32N6 | 8 kHz | external | 0.42 MiB | 1.22 MiB | 1000 ms | 1.54 | FAIL | compute,latency,offline_only |
| mp_senet_lite_16k | Alif Ensemble E3 | 16 kHz | onchip | 0.42 MiB | 2.42 MiB | 1000 ms | 6.86 | FAIL | compute,latency,offline_only |
| mp_senet_lite_16k | Infineon PSoC Edge E84 | 16 kHz | external | 0.42 MiB | 2.42 MiB | 1000 ms | 8.02 | FAIL | compute,latency,offline_only |
| mp_senet_lite_16k | NXP MCX N94 | 16 kHz | onchip | 0.42 MiB | 2.42 MiB | 1000 ms | 106.50 | FAIL | compute,latency,offline_only,sram |
| mp_senet_lite_16k | NXP i.MX RT700 | 16 kHz | external | 0.42 MiB | 2.42 MiB | 1000 ms | 44.72 | FAIL | compute,latency,offline_only |
| mp_senet_lite_16k | STM32L476RG | 16 kHz | onchip | 0.42 MiB | 2.42 MiB | 1000 ms | 398.61 | FAIL | compute,latency,offline_only,sram |
| mp_senet_lite_16k | STM32N6 | 16 kHz | external | 0.42 MiB | 2.42 MiB | 1000 ms | 3.84 | FAIL | compute,latency,offline_only |
