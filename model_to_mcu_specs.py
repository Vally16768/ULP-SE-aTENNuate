from __future__ import annotations


RUN_SPECS = [
    {
        "name": "atennuate_16k",
        "family": "atennuate",
        "sample_rate": 16000,
        "config": "experiments/model_to_mcu/atennuate_16k.toml",
        "run_dir": "runs/pesq_campaign/base/repo_baseline",
        "test_manifest": "dataset/voicebank-demand/16k/test.csv",
    },
    {
        "name": "atennuate_8k",
        "family": "atennuate",
        "sample_rate": 8000,
        "config": "experiments/model_to_mcu/atennuate_8k.toml",
        "run_dir": "runs/model_to_mcu/atennuate_8k",
        "test_manifest": "dataset/voicebank-demand/8k/test.csv",
    },
    {
        "name": "mp_senet_lite_16k",
        "family": "mp_senet_lite",
        "sample_rate": 16000,
        "config": "experiments/model_to_mcu/mp_senet_lite_16k.toml",
        "run_dir": "runs/mp_senet_lite_voicebank",
        "test_manifest": "dataset/voicebank-demand/16k/test.csv",
    },
    {
        "name": "mp_senet_lite_8k",
        "family": "mp_senet_lite",
        "sample_rate": 8000,
        "config": "experiments/model_to_mcu/mp_senet_lite_8k.toml",
        "run_dir": "runs/model_to_mcu/mp_senet_lite_8k",
        "test_manifest": "dataset/voicebank-demand/8k/test.csv",
    },
    {
        "name": "mp_senet_micro_16k",
        "family": "mp_senet_micro",
        "sample_rate": 16000,
        "config": "experiments/model_to_mcu/mp_senet_micro_16k.toml",
        "run_dir": "runs/model_to_mcu/mp_senet_micro_16k",
        "test_manifest": "dataset/voicebank-demand/16k/test.csv",
    },
    {
        "name": "mp_senet_micro_8k",
        "family": "mp_senet_micro",
        "sample_rate": 8000,
        "config": "experiments/model_to_mcu/mp_senet_micro_8k.toml",
        "run_dir": "runs/model_to_mcu/mp_senet_micro_8k",
        "test_manifest": "dataset/voicebank-demand/8k/test.csv",
    },
    {
        "name": "percepnet_class_16k",
        "family": "percepnet_class",
        "sample_rate": 16000,
        "config": "experiments/model_to_mcu/percepnet_class_16k.toml",
        "run_dir": "runs/model_to_mcu/percepnet_class_16k",
        "test_manifest": "dataset/voicebank-demand/16k/test.csv",
    },
    {
        "name": "percepnet_class_8k",
        "family": "percepnet_class",
        "sample_rate": 8000,
        "config": "experiments/model_to_mcu/percepnet_class_8k.toml",
        "run_dir": "runs/model_to_mcu/percepnet_class_8k",
        "test_manifest": "dataset/voicebank-demand/8k/test.csv",
    },
]


HARDWARE_COST_TIER = {
    "STM32L476RG": "low",
    "NXP MCX N94": "mid",
    "NXP i.MX RT700": "mid",
    "Infineon PSoC Edge E84": "high",
    "Alif Ensemble E3": "high",
    "STM32N6": "high",
}


def specs_for_sample_rate(sample_rate: int) -> list[dict[str, object]]:
    return [spec for spec in RUN_SPECS if int(spec["sample_rate"]) == int(sample_rate)]


def find_spec(*, family: str, sample_rate: int) -> dict[str, object]:
    for spec in RUN_SPECS:
        if str(spec["family"]) == str(family) and int(spec["sample_rate"]) == int(sample_rate):
            return spec
    raise KeyError(f"No run spec for family={family!r} sample_rate={sample_rate!r}")
