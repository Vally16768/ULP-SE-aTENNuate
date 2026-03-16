# MCU Low-Power Feasibility Report

This report compares 16 kHz and 8 kHz deployment profiles for the same model families and target MCUs. Evaluation mode: `stretch`.

## Final Recommendations

### Quality-first recommendation
Primary: `mp_senet_lite_8k_tiny_estimated` on `Alif Ensemble E3` at `8 kHz` -> `FAIL`.

### Efficiency-first recommendation
Primary: `mp_senet_lite_8k_tiny_estimated` on `Alif Ensemble E3` at `8 kHz` -> `FAIL`.

## Shortlist

| Category | Model | Hardware | Bandwidth | Verdict |
| --- | --- | --- | --- | --- |
| best classic MCU @16 kHz | none | - | - | - |
| best classic MCU @8 kHz | mp_senet_lite_8k_tiny_estimated | STM32L476RG | 8 kHz | FAIL |
| best MCU+NPU @16 kHz | none | - | - | - |
| best MCU+NPU @8 kHz | mp_senet_lite_8k_tiny_estimated | Alif Ensemble E3 | 8 kHz | FAIL |

## 16 kHz vs 8 kHz By Family And Hardware

_No rows._

## Alif Ensemble E3

### 16 kHz candidates
_No rows._

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_lite_8k_tiny_estimated | FAIL | degraded | 0.15 MiB | 2.05 MiB | 1008.0 | 0.00 | onchip | latency,offline_only |

## Infineon PSoC Edge E84

### 16 kHz candidates
_No rows._

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_lite_8k_tiny_estimated | FAIL | degraded | 0.15 MiB | 2.05 MiB | 1008.0 | 0.02 | external | latency,offline_only |

## NXP MCX N94

### 16 kHz candidates
_No rows._

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_lite_8k_tiny_estimated | FAIL | degraded | 0.15 MiB | 2.05 MiB | 1008.0 | 0.08 | onchip | latency,offline_only,sram |

## NXP i.MX RT700

### 16 kHz candidates
_No rows._

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_lite_8k_tiny_estimated | FAIL | degraded | 0.15 MiB | 2.05 MiB | 1008.0 | 0.03 | external | latency,offline_only |

## STM32L476RG

### 16 kHz candidates
_No rows._

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_lite_8k_tiny_estimated | FAIL | degraded | 0.15 MiB | 2.05 MiB | 1008.0 | 4.09 | onchip | compute,latency,offline_only,sram |

## STM32N6

### 16 kHz candidates
_No rows._

### 8 kHz candidates
| Model | Verdict | Quality | Flash | SRAM | Latency ms | RTF | Memory | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_lite_8k_tiny_estimated | FAIL | degraded | 0.15 MiB | 2.05 MiB | 1008.0 | 0.00 | external | latency,offline_only |

## Notes

- Strict mode means on-chip only. Stretch mode allows external flash and/or SRAM when the hardware profile exposes it.
- Porting-target mode evaluates feasibility with stretch rules, then narrows recommendations to streaming and causal candidates.
- 16 kHz and 8 kHz PESQ are not compared as absolute cross-band numbers. The report uses quality tiers and deployment gains instead.
- Estimated power is a coarse simulator output derived from core current, modeled engine load, configured NPU power, and optional external-memory penalty. It is useful for ranking, not for final power sign-off.
- aTENNuate and MP-SENet-lite current repo variants are treated as offline blocks, so algorithmic latency is a hard constraint.
