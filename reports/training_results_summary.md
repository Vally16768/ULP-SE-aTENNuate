# Training Results Summary

Current evaluated VoiceBank+DEMAND runs. Cross-sample-rate ranking uses `test delta_PESQ`, not raw `PESQ`.

| Rank | Model | SR | Best epoch | Val PESQ | Val STOI | Val SI-SDR | Test PESQ | Test STOI | Test SI-SDR | delta_PESQ | Stop |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `mp_senet_lite_16k` | 16 kHz | 114 | 2.8147 | 0.8982 | 16.6038 | 3.1107 | 0.9533 | 19.6763 | 1.1398 | `max_epochs` |
| 2 | `percepnet_class_16k` | 16 kHz | 48 | 2.2373 | 0.8663 | 14.0724 | 2.5052 | 0.9363 | 18.2460 | 0.5343 | `manual_stop` |
| 3 | `aTENNuate_16k` | 16 kHz | 78 | 1.9901 | 0.8618 | 12.6474 | 2.3550 | 0.9245 | 16.2249 | 0.3841 | `max_epochs` |
| 4 | `mp_senet_lite_8k_tiny` | 8 kHz | 120 | 2.8736 | 0.8419 | 13.8098 | 3.2998 | 0.9272 | 18.3153 | 0.3521 | `max_epochs` |
| 5 | `mp_senet_micro_16k` | 16 kHz | 118 | 1.8368 | 0.8471 | 12.8010 | 2.2017 | 0.9273 | 16.4799 | 0.2308 | `max_epochs` |

Leaders:
- Global normalized leader: `mp_senet_lite_16k` with `delta_PESQ 1.1398`.
- Best 16 kHz model: `mp_senet_lite_16k`.
- Best 8 kHz model: `mp_senet_lite_8k_tiny`.
