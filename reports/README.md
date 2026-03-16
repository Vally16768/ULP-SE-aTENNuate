# VoiceBank+DEMAND Results

This folder centralizes the current VoiceBank+DEMAND experiment outcomes and the MCU deployment screening derived from the trained checkpoints.

## Scope

The current summary covers:
- `aTENNuate_16k`
- `mp_senet_lite_16k`
- `mp_senet_micro_16k`
- `percepnet_class_16k`
- `mp_senet_lite_8k_tiny`

Cross-sample-rate ranking uses `delta_PESQ` on the matching test sample rate, not raw `PESQ`.

## Audio Ranking

| Rank | Model | SR | Best epoch | Val PESQ | Test PESQ | delta_PESQ | Test STOI | Test SI-SDR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `mp_senet_lite_16k` | 16 kHz | 114 | 2.8147 | 3.1107 | 1.1398 | 0.9533 | 19.6763 |
| 2 | `percepnet_class_16k` | 16 kHz | 48 | 2.2373 | 2.5052 | 0.5343 | 0.9363 | 18.2460 |
| 3 | `aTENNuate_16k` | 16 kHz | 78 | 1.9901 | 2.3550 | 0.3841 | 0.9245 | 16.2249 |
| 4 | `mp_senet_lite_8k_tiny` | 8 kHz | 120 | 2.8736 | 3.2998 | 0.3521 | 0.9272 | 18.3153 |
| 5 | `mp_senet_micro_16k` | 16 kHz | 118 | 1.8368 | 2.2017 | 0.2308 | 0.9273 | 16.4799 |

Current audio conclusions:
- Best overall model: `mp_senet_lite_16k`
- Best 16 kHz deployable MCU candidate: `percepnet_class_16k`
- Best 8 kHz experiment here: `mp_senet_lite_8k_tiny`, but it is still offline-only

## MCU Real-Time Shortlist

Filter used:
- `PASS`
- `on-chip`
- real-time according to the simulator
- estimated power under `50 mW`

| Model | MCU | Estimated power | RTF | Latency | Test PESQ | delta_PESQ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `percepnet_class_16k` | `Alif Ensemble E3` | 0.21 mW | 0.0058 | 20 ms | 2.5052 | 0.5343 |
| `mp_senet_micro_16k` | `Alif Ensemble E3` | 1.23 mW | 0.0314 | 20 ms | 2.2017 | 0.2308 |
| `percepnet_class_16k` | `NXP MCX N94` | 1.63 mW | 0.0587 | 20 ms | 2.5052 | 0.5343 |
| `mp_senet_micro_16k` | `NXP MCX N94` | 20.76 mW | 0.7631 | 20 ms | 2.2017 | 0.2308 |
| `percepnet_class_16k` | `STM32L476RG` | 22.58 mW | 0.8554 | 20 ms | 2.5052 | 0.5343 |

Current MCU conclusions:
- Best quality-first deployable pair: `percepnet_class_16k` on `Alif Ensemble E3`
- Best lower-cost deployable pair: `percepnet_class_16k` on `NXP MCX N94`
- Cheapest classic MCU that still passes: `percepnet_class_16k` on `STM32L476RG`

## Important Non-Deployable Results

These models currently do not pass the MCU real-time filter:
- `mp_senet_lite_16k`
- `aTENNuate_16k`
- `mp_senet_lite_8k_tiny`

For `mp_senet_lite_8k_tiny`, the blocker is not raw throughput. The blocker is structural:
- `algorithmic latency = 1008 ms`
- `streaming_mode = offline`

That model can process audio faster than real-time on a desktop machine, but it is not suitable for low-latency MCU audio deployment in its current form.

## Key Artifacts

Audio results:
- [training_results_summary.md](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/reports/training_results_summary.md)
- [training_results_summary.csv](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/reports/training_results_summary.csv)
- [training_results_summary.json](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/reports/training_results_summary.json)

Measured MCU screening:
- [mcu_feasibility_measured_current.md](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/reports/mcu_feasibility_measured_current.md)
- [mcu_feasibility_measured_current.csv](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/reports/mcu_feasibility_measured_current.csv)
- [mcu_feasibility_measured_current.json](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/reports/mcu_feasibility_measured_current.json)

Per-model test outputs:
- [mp_senet_lite_voicebank/test_eval.json](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/runs/mp_senet_lite_voicebank/test_eval.json)
- [repo_baseline/test_eval.json](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/runs/pesq_campaign/base/repo_baseline/test_eval.json)
- [percepnet_class_16k/test_eval.json](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/runs/model_to_mcu/percepnet_class_16k/test_eval.json)
- [mp_senet_micro_16k/test_eval.json](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/runs/model_to_mcu/mp_senet_micro_16k/test_eval.json)
- [mp_senet_lite_8k_tiny/test_eval.json](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/runs/model_to_mcu/mp_senet_lite_8k_tiny/test_eval.json)

Measured deployment profiles:
- [reports/measured_model_profiles](/c:/Users/E1554695/Desktop/projects/ULP-SE-aTENNuate/reports/measured_model_profiles)

## Notes

- Power figures are simulator estimates used for ranking and screening, not board-level measurements.
- The MCU result is attached to the trained checkpoint, so the same model keeps the same `PESQ`; only deployment feasibility changes across MCUs.
- If the next goal is embedded deployment, the strongest current direction is `percepnet_class_16k`, not `mp_senet_lite`.
