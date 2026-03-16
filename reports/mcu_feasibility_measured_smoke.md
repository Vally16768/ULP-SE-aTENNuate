# MCU Low-Power Feasibility Report

This report compares 16 kHz and 8 kHz deployment profiles for the same model families and target MCUs. Evaluation mode: `strict`.

## Final Recommendations

### Quality-first recommendation
Primary: `percepnet_class_16k` on `Alif Ensemble E3` at `16 kHz` -> `PASS`.

### Efficiency-first recommendation
Primary: `percepnet_class_16k` on `Alif Ensemble E3` at `16 kHz` -> `PASS`.

## Shortlist

| Category | Model | Hardware | Bandwidth | Verdict |
| --- | --- | --- | --- | --- |
| best classic MCU @16 kHz | percepnet_class_16k | STM32L476RG | 16 kHz | PASS |
| best classic MCU @8 kHz | none | - | - | - |
| best MCU+NPU @16 kHz | percepnet_class_16k | Alif Ensemble E3 | 16 kHz | PASS |
| best MCU+NPU @8 kHz | none | - | - | - |

## 16 kHz vs 8 kHz By Family And Hardware

_No rows._

## Alif Ensemble E3

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | unknown | 0.27 MiB | 0.01 MiB | 20.0 | 0.01 | onchip | - |
| mp_senet_micro_16k | PASS | unknown | 0.11 MiB | 0.24 MiB | 20.0 | 0.02 | onchip | - |
| atennuate_16k | FAIL | unknown | 0.90 MiB | 0.80 MiB | 1008.0 | 1.54 | onchip | compute,latency,offline_only |
| mp_senet_lite_16k | FAIL | unknown | 0.42 MiB | 23.87 MiB | 1000.0 | 2.48 | onchip | compute,latency,offline_only,sram |

### 8 kHz candidates
_No rows._

## Infineon PSoC Edge E84

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | FAIL | unknown | 0.27 MiB | 0.01 MiB | 20.0 | 0.01 | onchip | flash,flashless_onchip |
| mp_senet_micro_16k | FAIL | unknown | 0.11 MiB | 0.24 MiB | 20.0 | 0.05 | onchip | flash,flashless_onchip |
| atennuate_16k | FAIL | unknown | 0.90 MiB | 0.80 MiB | 1008.0 | 1.55 | onchip | compute,flash,flashless_onchip,latency,offline_only |
| mp_senet_lite_16k | FAIL | unknown | 0.42 MiB | 23.87 MiB | 1000.0 | 2.85 | onchip | compute,flash,flashless_onchip,latency,offline_only,sram |

### 8 kHz candidates
_No rows._

## NXP MCX N94

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | unknown | 0.27 MiB | 0.01 MiB | 20.0 | 0.06 | onchip | - |
| mp_senet_micro_16k | PASS | unknown | 0.11 MiB | 0.24 MiB | 20.0 | 0.49 | onchip | - |
| atennuate_16k | FAIL | unknown | 0.90 MiB | 0.80 MiB | 1008.0 | 4.15 | onchip | compute,latency,offline_only,sram |
| mp_senet_lite_16k | FAIL | unknown | 0.42 MiB | 23.87 MiB | 1000.0 | 49.51 | onchip | compute,latency,offline_only,sram |

### 8 kHz candidates
_No rows._

## NXP i.MX RT700

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | FAIL | unknown | 0.27 MiB | 0.01 MiB | 20.0 | 0.02 | onchip | flash,flashless_onchip |
| mp_senet_micro_16k | FAIL | unknown | 0.11 MiB | 0.24 MiB | 20.0 | 0.17 | onchip | flash,flashless_onchip |
| atennuate_16k | FAIL | unknown | 0.90 MiB | 0.80 MiB | 1008.0 | 1.28 | onchip | compute,flash,flashless_onchip,latency,offline_only |
| mp_senet_lite_16k | FAIL | unknown | 0.42 MiB | 23.87 MiB | 1000.0 | 19.31 | onchip | compute,flash,flashless_onchip,latency,offline_only,sram |

### 8 kHz candidates
_No rows._

## STM32L476RG

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | PASS | unknown | 0.27 MiB | 0.01 MiB | 20.0 | 0.86 | onchip | - |
| mp_senet_micro_16k | FAIL | unknown | 0.11 MiB | 0.24 MiB | 20.0 | 13.85 | onchip | compute,sram |
| atennuate_16k | FAIL | unknown | 0.90 MiB | 0.80 MiB | 1008.0 | 39.11 | onchip | compute,flash,latency,offline_only,sram |
| mp_senet_lite_16k | FAIL | unknown | 0.42 MiB | 23.87 MiB | 1000.0 | 231.96 | onchip | compute,latency,offline_only,sram |

### 8 kHz candidates
_No rows._

## STM32N6

### 16 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| percepnet_class_16k | FAIL | unknown | 0.27 MiB | 0.01 MiB | 20.0 | 0.00 | onchip | flash,flashless_onchip |
| mp_senet_micro_16k | FAIL | unknown | 0.11 MiB | 0.24 MiB | 20.0 | 0.01 | onchip | flash,flashless_onchip |
| atennuate_16k | FAIL | unknown | 0.90 MiB | 0.80 MiB | 1008.0 | 0.77 | onchip | flash,flashless_onchip,latency,offline_only |
| mp_senet_lite_16k | FAIL | unknown | 0.42 MiB | 23.87 MiB | 1000.0 | 1.24 | onchip | compute,flash,flashless_onchip,latency,offline_only,sram |

### 8 kHz candidates
_No rows._

## Notes

- Strict mode means on-chip only. Stretch mode allows external flash and/or SRAM when the hardware profile exposes it.
- Porting-target mode evaluates feasibility with stretch rules, then narrows recommendations to streaming and causal candidates.
- 16 kHz and 8 kHz PESQ are not compared as absolute cross-band numbers. The report uses quality tiers and deployment gains instead.
- Estimated power is a coarse simulator output derived from core current, modeled engine load, configured NPU power, and optional external-memory penalty. It is useful for ranking, not for final power sign-off.
- aTENNuate and MP-SENet-lite current repo variants are treated as offline blocks, so algorithmic latency is a hard constraint.
