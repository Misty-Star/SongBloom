"""Structure asset manifest/report merge helpers."""

from __future__ import annotations

import typing as tp


PREPARED_ASSET_KEYS = (
    "vocals_path",
    "no_vocals_path",
    "whisperx_json",
    "structure_json",
)


def compute_prepare_assets_status(record: dict, report: dict) -> str:
    errors = report.get("errors") or []
    has_any_asset = any(record.get(key) for key in PREPARED_ASSET_KEYS)
    if errors and has_any_asset:
        return "partial"
    if errors:
        return "error"
    return "ok"


def merge_structure_preparation_results(
    output_rows: tp.List[dict],
    report_rows: tp.List[dict],
    structure_items: tp.Sequence[dict],
    structure_reports: tp.Sequence[dict],
) -> None:
    output_by_id = {str(row["id"]): row for row in output_rows}
    report_by_id = {str(row["id"]): row for row in report_rows}

    for item in structure_items:
        sample_id = str(item["id"])
        output_row = output_by_id[sample_id]
        if item.get("structure_json"):
            output_row["structure_json"] = item["structure_json"]

    for structure_report in structure_reports:
        sample_id = str(structure_report["id"])
        report = report_by_id[sample_id]
        report["structure_status"] = structure_report.get("structure_status", "skipped")
        if structure_report.get("structure_json"):
            report["structure_json"] = structure_report["structure_json"]
        if structure_report.get("error"):
            report.setdefault("errors", []).append(
                {
                    "stage": "structure",
                    "error": structure_report["error"],
                }
            )

    for output_row in output_rows:
        sample_id = str(output_row["id"])
        report_by_id[sample_id]["status"] = compute_prepare_assets_status(
            record=output_row,
            report=report_by_id[sample_id],
        )
