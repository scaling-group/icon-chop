from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
MFC_CODE_ROOT = WORKSPACE_ROOT / "mean_field_control" / "icon_chop_mfc_code"
DEFAULT_PROGRAM = MFC_CODE_ROOT / "seed" / "initial_program.py"


def _bootstrap_mfc_runtime() -> None:
    for path in (MFC_CODE_ROOT / "eval", MFC_CODE_ROOT / "seed"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _positive_or_all(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"all", "full", "none", "null", "0"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'all'")
    return parsed


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    default_output = SCRIPT_DIR / "logs" / datetime.now().strftime("mfc_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the saved ICON-CHOP MFC post-evaluation. The defaults are the "
            "historical 100-operator, seed-0, 10x50-grid protocol."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing the 15 OOD MFC .h5 files.")
    parser.add_argument("--ckpt", required=True, help="MFC ICON checkpoint.")
    parser.add_argument("--program", default=str(DEFAULT_PROGRAM), help="Python program defining run_icon_chop(context).")
    parser.add_argument("--pattern", default="*.h5", help="Dataset glob relative to --data-dir.")
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete CUDA device.")
    parser.add_argument("--batch-size", type=int, default=1, help="Historical value is 1.")
    parser.add_argument("--num-samples-per-dataset", type=_positive_or_all, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-num", type=int, default=5)
    parser.add_argument("--max-len", type=_positive_or_all, default=100)
    parser.add_argument("--time-subsample", type=_positive_or_all, default=10)
    parser.add_argument("--space-subsample", type=_positive_or_all, default=50)
    parser.add_argument("--k-dim", type=int, default=2)
    parser.add_argument("--model-in-features", type=int, default=3)
    return parser.parse_args()


def _dataset_metadata(dataset_name: str) -> dict[str, str]:
    family = re.search(r"(gparam|rhoparam)", dataset_name)
    mode = re.search(r"(forward\d+)", dataset_name)
    return {
        "problem_family": family.group(1) if family else "",
        "mode": mode.group(1) if mode else "",
    }


def _rename_chain_keys(payload: dict[str, Any]) -> dict[str, Any]:
    renamed: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "score":
            key = "combined_score"
        key = key.replace("chain_", "agent_")
        renamed[key] = value
    return renamed


def _row_rank_key(row: dict[str, Any]) -> tuple[float, float, str]:
    return (-float(row["relative_improve"]), float(row["agent_rel_l2"]), str(row["sample_id"]))


def _summary(dataset_name: str, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    successful = [row for row in rows if bool(row["success"])]
    best = sorted(successful, key=_row_rank_key)[0] if successful else None
    metadata = _dataset_metadata(dataset_name)
    return {
        "dataset_name": dataset_name,
        **metadata,
        "num_samples": len(rows),
        "combined_score": float(metrics["combined_score"]),
        "baseline_rel_l2": float(metrics["baseline_rel_l2"]),
        "agent_rel_l2": float(metrics["agent_rel_l2"]),
        "mean_relative_improve": float(metrics["mean_relative_improve"]),
        "improve_ratio": float(metrics["improve_ratio"]),
        "regress_ratio": float(metrics["regress_ratio"]),
        "success_rate": float(metrics["success_rate"]),
        "failure_count": int(metrics["failure_count"]),
        "baseline_mse": float(metrics["baseline_mse"]),
        "agent_mse": float(metrics["agent_mse"]),
        "best_sample_id": str(best["sample_id"]) if best else "",
        "best_sample_available": best is not None,
        "best_sample_baseline_rel_l2": float(best["baseline_rel_l2"]) if best else None,
        "best_sample_agent_rel_l2": float(best["agent_rel_l2"]) if best else None,
        "best_sample_relative_improve": float(best["relative_improve"]) if best else None,
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    _bootstrap_mfc_runtime()

    from eval_core import evaluate_dataset_source, get_model, load_program_module, resolve_device
    from model.icon.wrapper import IconWrapper

    data_dir = _path(args.data_dir)
    if not any(data_dir.glob(args.pattern)) and any((data_dir / "eval_mfc").glob(args.pattern)):
        data_dir = (data_dir / "eval_mfc").resolve()
    ckpt_path = _path(args.ckpt)
    program_path = _path(args.program)
    output_dir = _path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    csv_dir = output_dir / "per_sample"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    for path, label in ((data_dir, "data directory"), (ckpt_path, "checkpoint"), (program_path, "program")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    data_sources = sorted(data_dir.glob(args.pattern))
    if not data_sources:
        raise FileNotFoundError(f"No datasets matched {data_dir / args.pattern}")

    device_arg = None if args.device == "auto" else args.device
    device = resolve_device(device_arg)
    module = load_program_module(str(program_path))
    if not hasattr(module, "run_icon_chop"):
        if hasattr(module, "run_icon_agent"):
            module.run_icon_chop = module.run_icon_agent
        else:
            raise AttributeError("Program must define run_icon_chop(context) or run_icon_agent(context)")
    model = get_model(str(ckpt_path), device=device, in_features=args.model_in_features)
    wrapper = IconWrapper(model)

    summaries: list[dict[str, Any]] = []
    print(f"MFC protocol: datasets={len(data_sources)} seed={args.seed} demos={args.demo_num} grid={args.time_subsample}x{args.space_subsample}")
    print(f"Device: {device}; checkpoint: {ckpt_path}")

    for index, data_path in enumerate(data_sources, start=1):
        print(f"[{index}/{len(data_sources)}] {data_path.name}", flush=True)
        result = evaluate_dataset_source(
            module=module,
            wrapper=wrapper,
            data_source=str(data_path),
            batch_size=args.batch_size,
            num_samples=args.num_samples_per_dataset,
            seed=args.seed,
            demo_num=args.demo_num,
            device=device,
            max_len=args.max_len,
            k_dim=args.k_dim,
            time_subsample=args.time_subsample,
            space_subsample=args.space_subsample,
        )
        dataset_name = str(result["dataset_name"])
        rows = [_rename_chain_keys(dict(row)) for row in result["rows"]]
        metrics = _rename_chain_keys(dict(result["metrics"]))
        summary = _summary(dataset_name, rows, metrics)
        dataset_payload = {"summary": summary, "rows": rows}
        dataset_json = metrics_dir / f"{dataset_name}.json"
        dataset_json.write_text(json.dumps(dataset_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_rows_csv(csv_dir / f"{dataset_name}.csv", rows)
        summaries.append(summary)
        print(
            f"  score={summary['combined_score']:.9f} "
            f"baseline={summary['baseline_rel_l2']:.9f} agent={summary['agent_rel_l2']:.9f}",
            flush=True,
        )
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()
        gc.collect()

    payload = {
        "protocol": "mfc_ood_500pt",
        "program": str(program_path),
        "ckpt": str(ckpt_path),
        "data_dir": str(data_dir),
        "pattern": args.pattern,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_samples_per_dataset": args.num_samples_per_dataset,
        "demo_num": args.demo_num,
        "seed": args.seed,
        "max_len": args.max_len,
        "time_subsample": args.time_subsample,
        "space_subsample": args.space_subsample,
        "k_dim": args.k_dim,
        "model_in_features": args.model_in_features,
        "dataset_count": len(data_sources),
        "dataset_files": [str(path) for path in data_sources],
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved MFC evaluation to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
