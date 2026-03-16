from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}
DEFAULT_HEADERS = ("name", "pesq", "csig", "cbak", "covl", "stoi", "sisdr", "ssnr", "dnsmos_ovr")
BASELINE_ALIASES = {
    "metricgan+": "metricgan_plus",
    "metricganplus": "metricgan_plus",
    "metricgan_plus": "metricgan_plus",
    "mp-senet": "mp_senet",
    "mpsenet": "mp_senet",
    "mp_senet": "mp_senet",
    "cmgan": "cmgan_small",
    "cmgan-small": "cmgan_small",
    "cmgan_small": "cmgan_small",
    "fullsubnet+": "fullsubnet_plus",
    "fullsubnetplus": "fullsubnet_plus",
    "fullsubnet_plus": "fullsubnet_plus",
    "fullsubnet": "fullsubnet_plus",
    "spectralgating": "spectral_gating",
    "spectral_gating": "spectral_gating",
    "attennuate": "atennuate",
    "a-tennuate": "atennuate",
    "a_tennuate": "atennuate",
}


def _column_to_index(ref: str) -> int:
    letters = "".join(character for character in ref if character.isalpha()).upper()
    index = 0
    for character in letters:
        index = index * 26 + (ord(character) - ord("A") + 1)
    return max(index - 1, 0)


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join((node.text or "") for node in cell.iterfind(".//a:t", XLSX_NS))
    value = cell.find("a:v", XLSX_NS)
    if value is None:
        return ""
    raw = value.text or ""
    if cell_type == "s":
        return shared_strings[int(raw)]
    return raw


def normalize_baseline_name(name: str | None) -> str:
    if not name:
        return ""
    compact = "".join(character.lower() for character in str(name) if character.isalnum() or character in {"+", "_", "-"})
    compact = compact.replace("-", "_")
    return BASELINE_ALIASES.get(compact, compact)


def _load_dense_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XLSX_NS):
                shared_strings.append("".join((node.text or "") for node in item.iterfind(".//a:t", XLSX_NS)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("p:Relationship", XLSX_NS)}
        first_sheet = workbook.find("a:sheets", XLSX_NS)[0]
        target = rel_map[first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
        sheet = ET.fromstring(archive.read(f"xl/{target}"))

        dense_rows: list[list[str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", XLSX_NS):
            indexed_values: dict[int, str] = {}
            for cell in row.findall("a:c", XLSX_NS):
                indexed_values[_column_to_index(cell.attrib.get("r", "A1"))] = _cell_value(cell, shared_strings)
            if not indexed_values:
                dense_rows.append([])
                continue
            max_index = max(indexed_values)
            dense_rows.append([indexed_values.get(index, "").strip() for index in range(max_index + 1)])
        return dense_rows


def load_classic_baselines(path: str | Path) -> list[dict[str, float | str | None]]:
    xlsx_path = Path(path)
    if not xlsx_path.exists():
        return []

    rows = _load_dense_rows(xlsx_path)
    if not rows:
        return []

    baselines: list[dict[str, float | str | None]] = []
    data_rows = rows
    for row_index, row in enumerate(data_rows, start=1):
        if not any(cell for cell in row):
            continue
        numeric_values = []
        for cell in row[1:] if len(row) > 1 else row:
            try:
                numeric_values.append(float(cell))
            except (TypeError, ValueError):
                pass
        if not numeric_values:
            continue

        padded = (row + [""] * len(DEFAULT_HEADERS))[: len(DEFAULT_HEADERS)]
        record: dict[str, float | str | None] = {
            "name": padded[0] or f"baseline_{row_index:02d}",
        }
        for header, value in zip(DEFAULT_HEADERS[1:], padded[1:]):
            try:
                record[header] = float(value)
            except (TypeError, ValueError):
                record[header] = None
        baselines.append(record)
    return baselines


def summarize_classic_baselines(path: str | Path) -> dict[str, object]:
    baselines = load_classic_baselines(path)
    pesq_rows = [row for row in baselines if row.get("pesq") is not None]
    if not pesq_rows:
        return {
            "path": str(Path(path)),
            "count": len(baselines),
            "top_pesq": None,
            "mean_pesq": None,
            "best_baseline": None,
        }
    best_baseline = max(pesq_rows, key=lambda row: float(row["pesq"]))  # type: ignore[arg-type]
    mean_pesq = sum(float(row["pesq"]) for row in pesq_rows) / len(pesq_rows)  # type: ignore[arg-type]
    return {
        "path": str(Path(path)),
        "count": len(baselines),
        "top_pesq": float(best_baseline["pesq"]),  # type: ignore[arg-type]
        "mean_pesq": mean_pesq,
        "best_baseline": best_baseline,
    }


def classic_pesq_index(path: str | Path) -> dict[str, float]:
    indexed: dict[str, float] = {}
    for row in load_classic_baselines(path):
        pesq = row.get("pesq")
        if pesq is None:
            continue
        normalized = normalize_baseline_name(str(row.get("name") or ""))
        if not normalized:
            continue
        indexed[normalized] = float(pesq)
    return indexed
