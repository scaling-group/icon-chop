#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_DATASET_COUNT = 15


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def validate_output_dir(output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    metrics_dir = output_dir / "metrics"
    artifacts_dir = output_dir / "artifacts"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"Missing metrics directory: {metrics_dir}")
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Missing artifacts directory: {artifacts_dir}")

    summary = _read_json(summary_path)
    dataset_count = int(summary.get("dataset_count", -1))
    if dataset_count != EXPECTED_DATASET_COUNT:
        raise AssertionError(f"Expected dataset_count={EXPECTED_DATASET_COUNT}, found {dataset_count}")
    if str(summary.get("device", "")) != "cuda":
        raise AssertionError(f"Expected summary device 'cuda', found {summary.get('device')!r}")

    summaries = summary.get("summaries")
    if not isinstance(summaries, list):
        raise TypeError("summary.json field 'summaries' must be a list")
    if len(summaries) != EXPECTED_DATASET_COUNT:
        raise AssertionError(f"Expected {EXPECTED_DATASET_COUNT} dataset summaries, found {len(summaries)}")

    failed_datasets: list[str] = []
    missing_artifacts: list[str] = []
    invalid_counts: list[str] = []
    for dataset_summary in summaries:
        if not isinstance(dataset_summary, dict):
            raise TypeError("Each dataset summary must be an object")
        dataset_name = str(dataset_summary.get("dataset_name", ""))
        if not dataset_name:
            raise AssertionError("Dataset summary is missing dataset_name")
        if float(dataset_summary.get("success_rate", 0.0)) <= 0.0:
            failed_datasets.append(f"{dataset_name}: success_rate={dataset_summary.get('success_rate')}")

        metrics_path = metrics_dir / f"{dataset_name}.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing dataset metrics JSON: {metrics_path}")

        metrics_payload = _read_json(metrics_path)
        artifact_records = metrics_payload.get("best_sample_artifacts")
        if not isinstance(artifact_records, list):
            raise TypeError(f"{metrics_path} field 'best_sample_artifacts' must be a list")
        if not artifact_records:
            failed_datasets.append(f"{dataset_name}: no best_sample_artifacts")
        if len(artifact_records) > 5:
            invalid_counts.append(f"{dataset_name}: {len(artifact_records)}")

        for record in artifact_records:
            if not isinstance(record, dict):
                raise TypeError(f"{metrics_path} contains a non-object artifact record")
            artifact_path = str(record.get("path", ""))
            if artifact_path and not Path(artifact_path).exists():
                missing_artifacts.append(artifact_path)

    if invalid_counts:
        raise AssertionError("Datasets with more than 5 artifact records: " + ", ".join(invalid_counts))
    if failed_datasets:
        raise AssertionError("Datasets without successful post-eval artifacts: " + ", ".join(failed_datasets))
    if missing_artifacts:
        raise FileNotFoundError("Missing artifact files: " + ", ".join(missing_artifacts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate best_program2 ICON-CHOP OOD post-eval outputs.")
    parser.add_argument("output_dir", type=Path, help="Post-eval output directory containing summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_output_dir(args.output_dir.resolve())
    print(f"Validated best_program2 ICON-CHOP OOD post-eval outputs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
