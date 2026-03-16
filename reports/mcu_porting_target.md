# MCU Low-Power Feasibility Report

This report compares 16 kHz and 8 kHz deployment profiles for the same model families and target MCUs. Evaluation mode: `porting-target`.

## Final Recommendations

### Quality-first recommendation
Primary: `percepnet_class_16k` on `Alif Ensemble E3` at `16 kHz` -> `PASS`.
Alternative: `mp_senet_micro_8k_target` on `Alif Ensemble E3` at `8 kHz` -> `PASS`.

### Efficiency-first recommendation
Primary: `rnnoise_class_8k` on `Alif Ensemble E3` at `8 kHz` -> `PASS`.
Alternative: `rnnoise_class_16k` on `Alif Ensemble E3` at `16 kHz` -> `PASS`.

## Porting Target

### Quality-first embedded target
Primary: `percepnet_class_16k` on `Alif Ensemble E3` at `16 kHz` -> `PASS`.
Alternative: `mp_senet_micro_8k_target` on `Alif Ensemble E3` at `8 kHz` -> `PASS`.

### Efficiency-first embedded target
Primary: `rnnoise_class_8k` on `Alif Ensemble E3` at `8 kHz` -> `PASS`.
Alternative: `rnnoise_class_16k` on `Alif Ensemble E3` at `16 kHz` -> `PASS`.

| Target | Model | Hardware | Bandwidth | Memory | Verdict |
| --- | --- | --- | --- | --- | --- |
| classic_mcu_target | spectral_gate_only_16k | STM32L476RG | 16 kHz | onchip | PASS |
| mcu_npu_target | percepnet_class_16k | Alif Ensemble E3 | 16 kHz | onchip | PASS |

### Current Repo Model Fit
| Family | Best candidate | Hardware | Bandwidth | Verdict | Reasons | Action |
| --- | --- | --- | --- | --- | --- | --- |
| atennuate | atennuate_16k | Alif Ensemble E3 | 16 kHz | FAIL | latency,offline_only | redesign_to_streaming_causal |
| mp_senet_lite | mp_senet_lite_16k | STM32L476RG | 16 kHz | FAIL | compute,latency,offline_only,sram | redesign_to_streaming_causal |

## Shortlist

| Category | Model | Hardware | Bandwidth | Verdict |
| --- | --- | --- | --- | --- |
| best classic MCU @16 kHz | spectral_gate_only_16k | STM32L476RG | 16 kHz | PASS |
| best classic MCU @8 kHz | spectral_gate_only_8k | STM32L476RG | 8 kHz | PASS |
| best MCU+NPU @16 kHz | percepnet_class_16k | Alif Ensemble E3 | 16 kHz | PASS |
| best MCU+NPU @8 kHz | mp_senet_micro_8k_target | Alif Ensemble E3 | 8 kHz | PASS |

## 16 kHz vs 8 kHz By Family And Hardware

| Family | Hardware | 16 kHz | 8 kHz | Compute 8/16 | SRAM 8/16 | Quality penalty | Deployment gain | Preference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atennuate | Alif Ensemble E3 | FAIL | FAIL | 0.51x | 0.50x | moderate | large | 16 kHz preferred |
| atennuate | Infineon PSoC Edge E84 | FAIL | FAIL | 0.51x | 0.50x | moderate | large | 16 kHz preferred |
| atennuate | NXP MCX N94 | FAIL | FAIL | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| atennuate | NXP i.MX RT700 | FAIL | FAIL | 0.49x | 0.50x | moderate | large | 16 kHz preferred |
| atennuate | STM32L476RG | FAIL | FAIL | 0.51x | 0.50x | moderate | large | 16 kHz preferred |
| atennuate | STM32N6 | FAIL | FAIL | 0.51x | 0.50x | moderate | large | 16 kHz preferred |
| mp_senet_lite | Alif Ensemble E3 | FAIL | FAIL | 0.40x | 0.50x | moderate | large | 16 kHz preferred |
| mp_senet_lite | Infineon PSoC Edge E84 | FAIL | FAIL | 0.40x | 0.50x | moderate | large | 16 kHz preferred |
| mp_senet_lite | NXP MCX N94 | FAIL | FAIL | 0.40x | 0.50x | moderate | large | 16 kHz preferred |
| mp_senet_lite | NXP i.MX RT700 | FAIL | FAIL | 0.40x | 0.50x | moderate | large | 16 kHz preferred |
| mp_senet_lite | STM32L476RG | FAIL | FAIL | 0.43x | 0.50x | moderate | large | 16 kHz preferred |
| mp_senet_lite | STM32N6 | FAIL | FAIL | 0.40x | 0.50x | moderate | large | 16 kHz preferred |
| mp_senet_micro | Alif Ensemble E3 | PASS | PASS | 0.48x | 0.68x | moderate | large | 8 kHz preferred |
| mp_senet_micro | Infineon PSoC Edge E84 | PASS | PASS | 0.49x | 0.68x | moderate | large | 8 kHz preferred |
| mp_senet_micro | NXP MCX N94 | PASS | PASS | 0.55x | 0.68x | moderate | medium | 8 kHz preferred |
| mp_senet_micro | NXP i.MX RT700 | PASS | PASS | 0.56x | 0.68x | moderate | medium | 8 kHz preferred |
| mp_senet_micro | STM32L476RG | FAIL | FAIL | 0.52x | 0.68x | moderate | medium | balanced |
| mp_senet_micro | STM32N6 | PASS | PASS | 0.48x | 0.68x | moderate | large | 8 kHz preferred |
| percepnet_class | Alif Ensemble E3 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| percepnet_class | Infineon PSoC Edge E84 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| percepnet_class | NXP MCX N94 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| percepnet_class | NXP i.MX RT700 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| percepnet_class | STM32L476RG | FAIL | FAIL | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| percepnet_class | STM32N6 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| rnnoise_class | Alif Ensemble E3 | PASS | PASS | 0.50x | 0.53x | moderate | large | 16 kHz preferred |
| rnnoise_class | Infineon PSoC Edge E84 | PASS | PASS | 0.50x | 0.53x | moderate | large | 16 kHz preferred |
| rnnoise_class | NXP MCX N94 | PASS | PASS | 0.50x | 0.53x | moderate | large | 16 kHz preferred |
| rnnoise_class | NXP i.MX RT700 | PASS | PASS | 0.50x | 0.53x | moderate | large | 16 kHz preferred |
| rnnoise_class | STM32L476RG | FAIL | PASS | 0.50x | 0.53x | moderate | large | 8 kHz preferred |
| rnnoise_class | STM32N6 | PASS | PASS | 0.50x | 0.53x | moderate | large | 16 kHz preferred |
| spectral_gate_only | Alif Ensemble E3 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| spectral_gate_only | Infineon PSoC Edge E84 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| spectral_gate_only | NXP MCX N94 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| spectral_gate_only | NXP i.MX RT700 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| spectral_gate_only | STM32L476RG | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |
| spectral_gate_only | STM32N6 | PASS | PASS | 0.50x | 0.50x | moderate | large | 16 kHz preferred |

## Alif Ensemble E3

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | high | 0.31 MiB | 0.20 MiB | 20.0 | 0.03 | onchip | - |
| rnnoise_class_16k | PASS | acceptable | 0.16 MiB | 0.07 MiB | 20.0 | 0.02 | onchip | - |
| spectral_gate_only_16k | PASS | acceptable | 0.05 MiB | 0.05 MiB | 32.0 | 0.02 | onchip | - |
| mp_senet_micro_16k_target | PASS | acceptable | 0.35 MiB | 0.26 MiB | 20.0 | 0.03 | onchip | - |
| atennuate_16k | FAIL | high | 0.90 MiB | 0.31 MiB | 1024.0 | 0.46 | onchip | latency,offline_only |
| mp_senet_lite_16k | FAIL | high | 0.42 MiB | 2.42 MiB | 1000.0 | 6.86 | onchip | compute,latency,offline_only |

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_micro_8k_target | PASS | acceptable | 0.27 MiB | 0.18 MiB | 20.0 | 0.01 | onchip | - |
| percepnet_class_8k | PASS | acceptable | 0.27 MiB | 0.10 MiB | 20.0 | 0.02 | onchip | - |
| rnnoise_class_8k | PASS | degraded | 0.14 MiB | 0.04 MiB | 20.0 | 0.01 | onchip | - |
| spectral_gate_only_8k | PASS | degraded | 0.05 MiB | 0.03 MiB | 32.0 | 0.01 | onchip | - |
| atennuate_8k_estimated | FAIL | acceptable | 0.90 MiB | 0.16 MiB | 1024.0 | 0.23 | onchip | latency,offline_only |
| mp_senet_lite_8k_estimated | FAIL | acceptable | 0.42 MiB | 1.22 MiB | 1000.0 | 2.74 | onchip | compute,latency,offline_only |

## Infineon PSoC Edge E84

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | high | 0.31 MiB | 0.20 MiB | 20.0 | 0.05 | external | - |
| rnnoise_class_16k | PASS | acceptable | 0.16 MiB | 0.07 MiB | 20.0 | 0.02 | external | - |
| spectral_gate_only_16k | PASS | acceptable | 0.05 MiB | 0.05 MiB | 32.0 | 0.03 | external | - |
| mp_senet_micro_16k_target | PASS | acceptable | 0.35 MiB | 0.26 MiB | 20.0 | 0.04 | external | - |
| atennuate_16k | FAIL | high | 0.90 MiB | 0.31 MiB | 1024.0 | 0.51 | external | latency,offline_only |
| mp_senet_lite_16k | FAIL | high | 0.42 MiB | 2.42 MiB | 1000.0 | 8.02 | external | compute,latency,offline_only |

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_micro_8k_target | PASS | acceptable | 0.27 MiB | 0.18 MiB | 20.0 | 0.02 | external | - |
| percepnet_class_8k | PASS | acceptable | 0.27 MiB | 0.10 MiB | 20.0 | 0.02 | external | - |
| rnnoise_class_8k | PASS | degraded | 0.14 MiB | 0.04 MiB | 20.0 | 0.01 | external | - |
| spectral_gate_only_8k | PASS | degraded | 0.05 MiB | 0.03 MiB | 32.0 | 0.01 | external | - |
| atennuate_8k_estimated | FAIL | acceptable | 0.90 MiB | 0.16 MiB | 1024.0 | 0.26 | external | latency,offline_only |
| mp_senet_lite_8k_estimated | FAIL | acceptable | 0.42 MiB | 1.22 MiB | 1000.0 | 3.24 | external | compute,latency,offline_only |

## NXP MCX N94

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | high | 0.31 MiB | 0.20 MiB | 20.0 | 0.52 | onchip | - |
| spectral_gate_only_16k | PASS | acceptable | 0.05 MiB | 0.05 MiB | 32.0 | 0.14 | onchip | - |
| mp_senet_micro_16k_target | PASS | acceptable | 0.35 MiB | 0.26 MiB | 20.0 | 0.17 | onchip | - |
| rnnoise_class_16k | PASS | acceptable | 0.16 MiB | 0.07 MiB | 20.0 | 0.26 | onchip | - |
| atennuate_16k | FAIL | high | 0.90 MiB | 0.31 MiB | 1024.0 | 2.25 | onchip | compute,latency,offline_only |
| mp_senet_lite_16k | FAIL | high | 0.42 MiB | 2.42 MiB | 1000.0 | 106.50 | onchip | compute,latency,offline_only,sram |

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_micro_8k_target | PASS | acceptable | 0.27 MiB | 0.18 MiB | 20.0 | 0.10 | onchip | - |
| percepnet_class_8k | PASS | acceptable | 0.27 MiB | 0.10 MiB | 20.0 | 0.26 | onchip | - |
| spectral_gate_only_8k | PASS | degraded | 0.05 MiB | 0.03 MiB | 32.0 | 0.07 | onchip | - |
| rnnoise_class_8k | PASS | degraded | 0.14 MiB | 0.04 MiB | 20.0 | 0.13 | onchip | - |
| atennuate_8k_estimated | FAIL | acceptable | 0.90 MiB | 0.16 MiB | 1024.0 | 1.12 | onchip | compute,latency,offline_only |
| mp_senet_lite_8k_estimated | FAIL | acceptable | 0.42 MiB | 1.22 MiB | 1000.0 | 42.72 | onchip | compute,latency,offline_only,sram |

## NXP i.MX RT700

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | high | 0.31 MiB | 0.20 MiB | 20.0 | 0.21 | external | - |
| spectral_gate_only_16k | PASS | acceptable | 0.05 MiB | 0.05 MiB | 32.0 | 0.05 | external | - |
| mp_senet_micro_16k_target | PASS | acceptable | 0.35 MiB | 0.26 MiB | 20.0 | 0.06 | external | - |
| rnnoise_class_16k | PASS | acceptable | 0.16 MiB | 0.07 MiB | 20.0 | 0.10 | external | - |
| atennuate_16k | FAIL | high | 0.90 MiB | 0.31 MiB | 1024.0 | 0.85 | external | latency,offline_only |
| mp_senet_lite_16k | FAIL | high | 0.42 MiB | 2.42 MiB | 1000.0 | 44.72 | external | compute,latency,offline_only |

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_micro_8k_target | PASS | acceptable | 0.27 MiB | 0.18 MiB | 20.0 | 0.04 | external | - |
| percepnet_class_8k | PASS | acceptable | 0.27 MiB | 0.10 MiB | 20.0 | 0.11 | external | - |
| spectral_gate_only_8k | PASS | degraded | 0.05 MiB | 0.03 MiB | 32.0 | 0.03 | external | - |
| rnnoise_class_8k | PASS | degraded | 0.14 MiB | 0.04 MiB | 20.0 | 0.05 | external | - |
| atennuate_8k_estimated | FAIL | acceptable | 0.90 MiB | 0.16 MiB | 1024.0 | 0.42 | external | latency,offline_only |
| mp_senet_lite_8k_estimated | FAIL | acceptable | 0.42 MiB | 1.22 MiB | 1000.0 | 17.92 | external | compute,latency,offline_only |

## STM32L476RG

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spectral_gate_only_16k | PASS | acceptable | 0.05 MiB | 0.05 MiB | 32.0 | 0.61 | onchip | - |
| percepnet_class_16k | FAIL | high | 0.31 MiB | 0.20 MiB | 20.0 | 4.72 | onchip | compute,sram |
| atennuate_16k | FAIL | high | 0.90 MiB | 0.31 MiB | 1024.0 | 14.39 | onchip | compute,flash,latency,offline_only,sram |
| mp_senet_lite_16k | FAIL | high | 0.42 MiB | 2.42 MiB | 1000.0 | 398.61 | onchip | compute,latency,offline_only,sram |
| rnnoise_class_16k | FAIL | acceptable | 0.16 MiB | 0.07 MiB | 20.0 | 1.97 | onchip | compute |
| mp_senet_micro_16k_target | FAIL | acceptable | 0.35 MiB | 0.26 MiB | 20.0 | 2.81 | onchip | compute,sram |

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spectral_gate_only_8k | PASS | degraded | 0.05 MiB | 0.03 MiB | 32.0 | 0.31 | onchip | - |
| rnnoise_class_8k | PASS | degraded | 0.14 MiB | 0.04 MiB | 20.0 | 0.99 | onchip | - |
| mp_senet_micro_8k_target | FAIL | acceptable | 0.27 MiB | 0.18 MiB | 20.0 | 1.47 | onchip | compute,sram |
| percepnet_class_8k | FAIL | acceptable | 0.27 MiB | 0.10 MiB | 20.0 | 2.36 | onchip | compute,sram |
| atennuate_8k_estimated | FAIL | acceptable | 0.90 MiB | 0.16 MiB | 1024.0 | 7.29 | onchip | compute,flash,latency,offline_only,sram |
| mp_senet_lite_8k_estimated | FAIL | acceptable | 0.42 MiB | 1.22 MiB | 1000.0 | 169.44 | onchip | compute,latency,offline_only,sram |

## STM32N6

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | high | 0.31 MiB | 0.20 MiB | 20.0 | 0.02 | external | - |
| rnnoise_class_16k | PASS | acceptable | 0.16 MiB | 0.07 MiB | 20.0 | 0.01 | external | - |
| spectral_gate_only_16k | PASS | acceptable | 0.05 MiB | 0.05 MiB | 32.0 | 0.01 | external | - |
| mp_senet_micro_16k_target | PASS | acceptable | 0.35 MiB | 0.26 MiB | 20.0 | 0.02 | external | - |
| atennuate_16k | FAIL | high | 0.90 MiB | 0.31 MiB | 1024.0 | 0.26 | external | latency,offline_only |
| mp_senet_lite_16k | FAIL | high | 0.42 MiB | 2.42 MiB | 1000.0 | 3.84 | external | compute,latency,offline_only |

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_micro_8k_target | PASS | acceptable | 0.27 MiB | 0.18 MiB | 20.0 | 0.01 | external | - |
| percepnet_class_8k | PASS | acceptable | 0.27 MiB | 0.10 MiB | 20.0 | 0.01 | external | - |
| rnnoise_class_8k | PASS | degraded | 0.14 MiB | 0.04 MiB | 20.0 | 0.01 | external | - |
| spectral_gate_only_8k | PASS | degraded | 0.05 MiB | 0.03 MiB | 32.0 | 0.01 | external | - |
| atennuate_8k_estimated | FAIL | acceptable | 0.90 MiB | 0.16 MiB | 1024.0 | 0.13 | external | latency,offline_only |
| mp_senet_lite_8k_estimated | FAIL | acceptable | 0.42 MiB | 1.22 MiB | 1000.0 | 1.54 | external | compute,latency,offline_only |

## Notes

- Strict mode means on-chip only. Stretch mode allows external flash and/or SRAM when the hardware profile exposes it.
- Porting-target mode evaluates feasibility with stretch rules, then narrows recommendations to streaming and causal candidates.
- 16 kHz and 8 kHz PESQ are not compared as absolute cross-band numbers. The report uses quality tiers and deployment gains instead.
- Estimated power is a coarse simulator output derived from core current, modeled engine load, configured NPU power, and optional external-memory penalty. It is useful for ranking, not for final power sign-off.
- aTENNuate and MP-SENet-lite current repo variants are treated as offline blocks, so algorithmic latency is a hard constraint.
