# MP-SENet-micro Target

This document captures a deployment-oriented redesign target for the current `MP-SENet-lite` family. It is a sizing and architecture target for embedded deployment, not a trained or validated acoustic result.

## Why A New Target Is Needed

The current repo model `MP-SENet-lite` performs well on VoiceBank, but it is not MCU-friendly in its present form:

- best validated local result: `PESQ 2.8147`, `STOI 0.8982`, `SI-SDR 16.6038 dB`
- reference: `runs/mp_senet_lite_voicebank/summary.json`
- deployment verdict in the MCU simulator: `FAIL` on all low-power MCU targets, even at estimated `8 kHz`

The blockers are structural:

- offline inference
- `~1 s` algorithmic latency
- high activation footprint
- FFT + TF-complex pipeline cost

## Target Architecture

`MP-SENet-micro` is the proposed embedded redesign target:

- causal streaming STFT front-end
- short frames only:
  - `16 kHz`: `20 ms` frame, `10 ms` hop
  - `8 kHz`: `20 ms` frame, `10 ms` hop
- depthwise-separable conv encoder/decoder
- tiny causal recurrent bottleneck
- mask-based spectral decoder
- noisy phase reuse instead of a heavy explicit phase branch
- int8 weights as default deployment assumption
- no attention blocks
- no offline block processing

This keeps the family resemblance to `MP-SENet` while removing the exact parts that break MCU deployment.

## Budgeted Profiles

### 16 kHz target

- profile: `profiles/models/mp_senet_micro_16k_target.json`
- flash: `0.35 MiB`
- SRAM peak: `0.26 MiB`
- latency: `20 ms`
- quality tier: `acceptable`

### 8 kHz target

- profile: `profiles/models/mp_senet_micro_8k_target.json`
- flash: `0.27 MiB`
- SRAM peak: `0.18 MiB`
- latency: `20 ms`
- quality tier: `acceptable`

## Simulator Verdicts

Results below come from:

- `reports/mcu_feasibility.md`
- `reports/mcu_feasibility_stretch.md`
- `reports/mcu_porting_target.md`

### Strict mode

| Model | Hardware | Verdict | Flash | SRAM | Latency | RTF | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mp_senet_micro_16k_target | Alif Ensemble E3 | PASS | 0.35 MiB | 0.26 MiB | 20 ms | 0.03 | best on-chip fit |
| mp_senet_micro_16k_target | NXP MCX N94 | PASS | 0.35 MiB | 0.26 MiB | 20 ms | 0.17 | realistic compact MCU+NPU target |
| mp_senet_micro_16k_target | STM32L476RG | FAIL | 0.35 MiB | 0.26 MiB | 20 ms | 2.81 | compute + SRAM |
| mp_senet_micro_8k_target | Alif Ensemble E3 | PASS | 0.27 MiB | 0.18 MiB | 20 ms | 0.01 | easiest on-chip fit |
| mp_senet_micro_8k_target | NXP MCX N94 | PASS | 0.27 MiB | 0.18 MiB | 20 ms | 0.10 | good low-power target |
| mp_senet_micro_8k_target | STM32L476RG | FAIL | 0.27 MiB | 0.18 MiB | 20 ms | 1.47 | still too slow and SRAM-limited |

### Stretch mode

With external memory allowed, the target also passes on:

- `Infineon PSoC Edge E84`
- `NXP i.MX RT700`
- `STM32N6`

But the primary targets remain:

- quality-first MCU+NPU: `MP-SENet-micro 16 kHz` on `Alif Ensemble E3`
- compact practical MCU+NPU: `MP-SENet-micro 8 kHz` or `16 kHz` on `NXP MCX N94`

## Conclusions

- Quantization alone does not rescue the current `MP-SENet-lite`.
- A structural redesign does.
- `STM32L476RG` is still too small for this target, even at `8 kHz`.
- `NXP MCX N94` is the smallest realistic on-chip target for `MP-SENet-micro`.
- `Alif Ensemble E3` is the safest and highest-margin target.

## Recommended Next Step

Implement a new model family in code:

- `mp_senet_micro`
- streaming only
- causal only
- `8 kHz` and `16 kHz` configs
- deployment target:
  - `MCX N94` for compact product
  - `Alif Ensemble E3` for safer quality-first product
