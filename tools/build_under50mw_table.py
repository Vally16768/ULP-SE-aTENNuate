from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRETCH_JSON = ROOT / "reports" / "mcu_feasibility_stretch.json"
OUT_MD = ROOT / "reports" / "embedded_under50mw_comparison.md"
OUT_CSV = ROOT / "reports" / "embedded_under50mw_comparison.csv"


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def load_rows() -> list[dict]:
    payload = json.loads(STRETCH_JSON.read_text(encoding="utf-8"))
    return payload["rows"]


def main() -> None:
    rows = load_rows()

    single_chip = [row for row in rows if row["verdict"] == "PASS" and row["estimated_energy_per_second_mj"] <= 50.0]
    onchip = [row for row in single_chip if row["memory_mode"] == "onchip"]
    external = [row for row in single_chip if row["memory_mode"] == "external"]

    def sort_key(row: dict) -> tuple:
        quality_rank = {"high": 3, "acceptable": 2, "degraded": 1, "low": 0}.get(row["quality_tier"], -1)
        return (
            1 if row["memory_mode"] == "onchip" else 0,
            quality_rank,
            row["sample_rate"],
            -row["estimated_energy_per_second_mj"],
            -row["real_time_factor"],
        )

    single_chip_sorted = sorted(single_chip, key=sort_key, reverse=True)
    onchip_sorted = sorted(onchip, key=sort_key, reverse=True)
    external_sorted = sorted(external, key=sort_key, reverse=True)

    top_compact = sorted(
        [row for row in onchip_sorted if row["hardware"] in {"NXP MCX N94", "Alif Ensemble E3", "STM32L476RG"}],
        key=lambda row: (
            {"high": 3, "acceptable": 2, "degraded": 1, "low": 0}.get(row["quality_tier"], -1),
            row["sample_rate"],
            -row["estimated_energy_per_second_mj"],
            -row["real_time_factor"],
        ),
        reverse=True,
    )[:8]

    current_repo = [
        row
        for row in rows
        if row["family"] in {"atennuate", "mp_senet_lite"}
    ]
    current_repo_sorted = sorted(
        current_repo,
        key=lambda row: (row["family"], row["sample_rate"], row["hardware"]),
    )

    headers = [
        "Model",
        "Hardware",
        "Bandwidth",
        "Quality",
        "Memory",
        "Flash",
        "SRAM",
        "Latency",
        "RTF",
        "Power",
        "Verdict",
    ]

    def to_table_rows(items: list[dict]) -> list[list[str]]:
        out: list[list[str]] = []
        for row in items:
            out.append(
                [
                    row["model"],
                    row["hardware"],
                    f"{row['sample_rate'] // 1000} kHz",
                    row["quality_tier"],
                    row["memory_mode"],
                    f"{mib(row['flash_bytes']):.2f} MiB",
                    f"{mib(row['sram_peak_bytes']):.2f} MiB",
                    f"{row['algorithmic_latency_ms']:.0f} ms",
                    f"{row['real_time_factor']:.2f}",
                    f"{row['estimated_energy_per_second_mj']:.2f} mW",
                    row["verdict"],
                ]
            )
        return out

    repo_headers = [
        "Model",
        "Hardware",
        "Bandwidth",
        "Memory",
        "Flash",
        "SRAM",
        "Latency",
        "RTF",
        "Verdict",
        "Reasons",
    ]
    repo_rows = []
    for row in current_repo_sorted:
        repo_rows.append(
            [
                row["model"],
                row["hardware"],
                f"{row['sample_rate'] // 1000} kHz",
                row["memory_mode"],
                f"{mib(row['flash_bytes']):.2f} MiB",
                f"{mib(row['sram_peak_bytes']):.2f} MiB",
                f"{row['algorithmic_latency_ms']:.0f} ms",
                f"{row['real_time_factor']:.2f}",
                row["verdict"],
                ",".join(row["reasons"]) or "-",
            ]
        )

    quality_first = next((row for row in onchip_sorted if row["quality_tier"] == "high"), onchip_sorted[0] if onchip_sorted else None)
    efficiency_first = min(onchip_sorted, key=lambda row: row["estimated_energy_per_second_mj"], default=None)

    md_lines = [
        "# Embedded Real-Time Under-50mW Comparison",
        "",
        "This report centralizes all currently explored deployment ideas under the product constraint:",
        "",
        "- real-time",
        "- single independent chip",
        "- preferably under 50 mW",
        "",
        "The data comes from `reports/mcu_feasibility_stretch.json` and includes both on-chip and external-memory single-chip options.",
        "Power values are simulator estimates, not oscilloscope or board measurements.",
        "",
        "## Primary Conclusions",
        "",
    ]

    if quality_first:
        md_lines.append(
            f"- Quality-first single-chip option: `{quality_first['model']}` on `{quality_first['hardware']}` at `{quality_first['sample_rate'] // 1000} kHz`, `{quality_first['estimated_energy_per_second_mj']:.2f} mW`, `{quality_first['memory_mode']}`."
        )
    if efficiency_first:
        md_lines.append(
            f"- Efficiency-first single-chip option: `{efficiency_first['model']}` on `{efficiency_first['hardware']}` at `{efficiency_first['sample_rate'] // 1000} kHz`, `{efficiency_first['estimated_energy_per_second_mj']:.2f} mW`, `{efficiency_first['memory_mode']}`."
        )
    md_lines.extend(
        [
            "- `STM32L476RG` remains suitable only for very small paths such as `spectral_gate_only` and `rnnoise_class_8k`.",
            "- `MP-SENet-lite` and `aTENNuate` current repo variants are not viable for this budget because they remain offline and/or too compute-heavy.",
            "- `MP-SENet-micro` is the first redesign target that becomes realistic on `NXP MCX N94` and `Alif Ensemble E3`.",
            "",
            "## Best On-Chip Candidates",
            "",
            fmt_table(headers, to_table_rows(top_compact)),
            "",
            "## All Single-Chip PASS Candidates Under 50 mW",
            "",
            fmt_table(headers, to_table_rows(single_chip_sorted)),
            "",
            "## On-Chip Only PASS Candidates Under 50 mW",
            "",
            fmt_table(headers, to_table_rows(onchip_sorted)),
            "",
            "## External-Memory Single-Chip PASS Candidates Under 50 mW",
            "",
            fmt_table(headers, to_table_rows(external_sorted)),
            "",
            "## Current Repo Models: Why They Still Fail",
            "",
            fmt_table(repo_headers, repo_rows),
            "",
        ]
    )

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", *headers])
        for section_name, items in (
            ("best_onchip", top_compact),
            ("all_under_50mw", single_chip_sorted),
            ("onchip_under_50mw", onchip_sorted),
            ("external_under_50mw", external_sorted),
        ):
            for row in to_table_rows(items):
                writer.writerow([section_name, *row])
        writer.writerow([])
        writer.writerow(["repo_models", *repo_headers])
        for row in repo_rows:
            writer.writerow(["repo_models", *row])

    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
