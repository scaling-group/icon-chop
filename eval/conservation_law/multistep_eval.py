"""Evaluate raw ICON and ICON-Chop on autoregressive conservation-law rollouts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from icon_model import IconWrapper, load_icon_model


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = WORKSPACE_ROOT.parent / "chop_data" / "eval" / "eval_conservation"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "ckpt" / "conservation_law" / "conservation.ckpt"
DEFAULT_CHOP_PROGRAM = WORKSPACE_ROOT / "conservation_law" / "icon_chop_weno_code" / "initial_program.py"
DEMO_NUM = 5
METHOD_ORDER = ("raw_icon", "manual_linear_prompt_scale", "icon_chop")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {name!r} was requested, but CUDA is unavailable")
    return torch.device(name)


def _unique_output_dir(base: Path, run_name: str | None) -> Path:
    stem = run_name or datetime.now().strftime("ms10_seed0_%Y%m%d_%H%M%S")
    candidate = base / stem
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _load_chop_runner(program_path: Path) -> Callable[[dict[str, Any]], torch.Tensor]:
    module_name = f"conservation_law_icon_chop_{abs(hash(str(program_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ICON-Chop program: {program_path}")
    program_dir = str(program_path.parent)
    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    entrypoint = getattr(module, "run_icon_chop", None) or getattr(module, "run_icon_agent", None)
    if entrypoint is None:
        raise AttributeError(f"{program_path} does not export run_icon_chop or run_icon_agent")
    return entrypoint


def _data_dict(template: dict[str, torch.Tensor], state: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        **template,
        "query_input_vals": state[:, None].contiguous(),
    }


def _expand_batch_stat(stat: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    stat = stat.reshape(-1)
    if value.shape[0] == stat.shape[0]:
        expanded = stat
    elif value.shape[0] % stat.shape[0] == 0:
        expanded = stat.repeat_interleave(value.shape[0] // stat.shape[0])
    else:
        raise ValueError("Batch size mismatch for prompt transform statistics")
    return expanded.to(value.device).reshape(-1, *([1] * (value.ndim - 1)))


class PromptLinearMinMaxTransform:
    """Historical WENO manual baseline: scale all prompt values to [-1, 1]."""

    def __init__(self, data: dict[str, torch.Tensor]):
        batch_size = data["demo_input_vals"].shape[0]
        values = torch.cat(
            [
                data["demo_input_vals"].reshape(batch_size, -1),
                data["demo_target_vals"].reshape(batch_size, -1),
                data["query_input_vals"].reshape(batch_size, -1),
            ],
            dim=1,
        )
        self.v_min = values.min(dim=1).values
        scale = values.max(dim=1).values - self.v_min
        self.scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        v_min = _expand_batch_stat(self.v_min, value)
        scale = _expand_batch_stat(self.scale, value)
        return ((value - v_min) / scale) * 2.0 - 1.0

    def backward(self, value: torch.Tensor) -> torch.Tensor:
        v_min = _expand_batch_stat(self.v_min, value)
        scale = _expand_batch_stat(self.scale, value)
        return ((value + 1.0) * 0.5) * scale + v_min


def _metric_arrays(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, np.ndarray]:
    difference = target - prediction
    reduce_dims = tuple(range(1, target.ndim))
    mse = torch.mean(difference.square(), dim=reduce_dims)
    l1 = torch.mean(difference.abs(), dim=reduce_dims)
    flat_difference = difference.reshape(difference.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    numerator = torch.linalg.vector_norm(flat_difference, dim=1)
    denominator = torch.linalg.vector_norm(flat_target, dim=1)
    rel_l2 = torch.where(
        denominator != 0,
        numerator / denominator,
        torch.where(numerator == 0, torch.zeros_like(numerator), torch.full_like(numerator, float("inf"))),
    )
    return {
        "mse": mse.detach().cpu().numpy().astype(np.float64),
        "rel_l2": rel_l2.detach().cpu().numpy().astype(np.float64),
        "l1": l1.detach().cpu().numpy().astype(np.float64),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else -1.0
    return (baseline - candidate) / baseline


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    summaries = []
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        horizons = sorted({int(row["horizon"]) for row in method_rows})
        for horizon in horizons:
            selected = [row for row in method_rows if int(row["horizon"]) == horizon]
            mean_rel = float(np.mean([float(row["rel_l2"]) for row in selected]))
            mean_raw_rel = float(np.mean([float(row["raw_icon_rel_l2"]) for row in selected]))
            summaries.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "horizon": horizon,
                    "num_samples": len(selected),
                    "combined_score": _ratio(mean_raw_rel, mean_rel),
                    "mean_rel_l2": mean_rel,
                    "mean_raw_icon_rel_l2": mean_raw_rel,
                    "mean_mse": float(np.mean([float(row["mse"]) for row in selected])),
                    "mean_raw_icon_mse": float(np.mean([float(row["raw_icon_mse"]) for row in selected])),
                    "mean_l1": float(np.mean([float(row["l1"]) for row in selected])),
                    "mean_raw_icon_l1": float(np.mean([float(row["raw_icon_l1"]) for row in selected])),
                    "win_rate_vs_raw_icon": float(
                        np.mean([float(row["rel_l2"]) <= float(row["raw_icon_rel_l2"]) for row in selected])
                    ),
                    "mean_relative_improve_vs_raw_icon": float(
                        np.mean([float(row["relative_improve"]) for row in selected])
                    ),
                }
            )
    return summaries


def _overall_summary(rows: list[dict[str, Any]], dataset_name: str) -> list[dict[str, Any]]:
    output = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        mean_rel = float(np.mean([float(row["rel_l2"]) for row in selected]))
        mean_raw_rel = float(np.mean([float(row["raw_icon_rel_l2"]) for row in selected]))
        output.append(
            {
                "dataset": dataset_name,
                "method": method,
                "num_predictions": len(selected),
                "combined_score": _ratio(mean_raw_rel, mean_rel),
                "mean_rel_l2": mean_rel,
                "mean_raw_icon_rel_l2": mean_raw_rel,
                "mean_mse": float(np.mean([float(row["mse"]) for row in selected])),
                "mean_l1": float(np.mean([float(row["l1"]) for row in selected])),
                "win_rate_vs_raw_icon": float(
                    np.mean([float(row["rel_l2"]) <= float(row["raw_icon_rel_l2"]) for row in selected])
                ),
                "mean_relative_improve_vs_raw_icon": float(
                    np.mean([float(row["relative_improve"]) for row in selected])
                ),
            }
        )
    return output


def evaluate_dataset(
    data_path: Path,
    wrapper: IconWrapper,
    chop_runner: Callable[[dict[str, Any]], torch.Tensor],
    device: torch.device,
    num_samples: int,
    batch_size: int,
    rollout_steps: int,
    seed: int,
    output_dir: Path,
    save_predictions: bool,
    include_manual: bool,
    log: Callable[[str], None],
) -> dict[str, Any]:
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    sequences = payload["data"].float()
    if sequences.ndim != 4:
        raise ValueError(f"Expected [N,T,P,C] data in {data_path}, got {tuple(sequences.shape)}")
    total, sequence_length, num_points, channels = sequences.shape
    available_steps = sequence_length - DEMO_NUM - 1
    if rollout_steps > available_steps:
        raise ValueError(
            f"{data_path.name} supports at most {available_steps} rollout steps with demo_num={DEMO_NUM}, "
            f"but {rollout_steps} were requested"
        )
    indices = np.random.RandomState(seed).permutation(total)[: min(num_samples, total)]
    coords = torch.linspace(0, 1, num_points, dtype=torch.float32).reshape(1, 1, num_points, 1)
    rows: list[dict[str, Any]] = []
    raw_rollouts: list[torch.Tensor] = []
    manual_rollouts: list[torch.Tensor] = []
    chop_rollouts: list[torch.Tensor] = []
    target_rollouts: list[torch.Tensor] = []

    log(f"dataset={data_path.name} shape={tuple(sequences.shape)} samples={len(indices)} steps={rollout_steps}")
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        sequence = sequences[batch_indices].to(device)
        current_batch_size = len(batch_indices)
        batch_coords = coords.expand(current_batch_size, DEMO_NUM, num_points, 1).to(device)
        query_coords = coords.expand(current_batch_size, 1, num_points, 1).to(device)
        template = {
            "demo_input_coords": batch_coords,
            "demo_input_vals": sequence[:, :DEMO_NUM].contiguous(),
            "demo_target_coords": batch_coords,
            "demo_target_vals": sequence[:, 1 : DEMO_NUM + 1].contiguous(),
            "query_input_coords": query_coords,
            "query_target_coords": query_coords,
        }
        raw_state = sequence[:, DEMO_NUM].contiguous()
        manual_state = raw_state.clone()
        chop_state = raw_state.clone()
        batch_raw_predictions = []
        batch_manual_predictions = []
        batch_chop_predictions = []
        batch_targets = []

        for horizon in range(1, rollout_steps + 1):
            target = sequence[:, DEMO_NUM + horizon]
            raw_prediction = wrapper.predict(data=_data_dict(template, raw_state))
            manual_prediction = None
            if include_manual:
                manual_data = _data_dict(template, manual_state)
                manual_prediction = wrapper.predict(
                    transform=PromptLinearMinMaxTransform(manual_data), data=manual_data
                )
            chop_data = _data_dict(template, chop_state)
            chop_prediction = chop_runner({"data": chop_data, "model": wrapper})
            predictions = [raw_prediction, chop_prediction]
            if manual_prediction is not None:
                predictions.append(manual_prediction)
            if any(prediction.shape != target.shape for prediction in predictions):
                raise ValueError(
                    f"Prediction shape mismatch at horizon {horizon}: target={tuple(target.shape)}, "
                    f"predictions={[tuple(prediction.shape) for prediction in predictions]}"
                )
            raw_metrics = _metric_arrays(target, raw_prediction)
            manual_metrics = _metric_arrays(target, manual_prediction) if manual_prediction is not None else None
            chop_metrics = _metric_arrays(target, chop_prediction)
            for local_index, source_index in enumerate(batch_indices):
                raw_values = {key: float(value[local_index]) for key, value in raw_metrics.items()}
                chop_values = {key: float(value[local_index]) for key, value in chop_metrics.items()}
                common = {
                    "dataset": data_path.stem,
                    "sample_id": f"sample_{int(source_index)}",
                    "source_index": int(source_index),
                    "horizon": horizon,
                    "raw_icon_mse": raw_values["mse"],
                    "raw_icon_rel_l2": raw_values["rel_l2"],
                    "raw_icon_l1": raw_values["l1"],
                    "success": True,
                    "error": "",
                }
                candidates = [("raw_icon", raw_values)]
                if manual_metrics is not None:
                    candidates.append(
                        (
                            "manual_linear_prompt_scale",
                            {key: float(value[local_index]) for key, value in manual_metrics.items()},
                        )
                    )
                candidates.append(("icon_chop", chop_values))
                for method, values in candidates:
                    rows.append(
                        {
                            "method": method,
                            **common,
                            "mse": values["mse"],
                            "rel_l2": values["rel_l2"],
                            "l1": values["l1"],
                            "improvement_ratio": _ratio(raw_values["rel_l2"], values["rel_l2"]),
                            "relative_improve": _relative_improvement(raw_values["rel_l2"], values["rel_l2"]),
                        }
                    )
            raw_state = raw_prediction
            if manual_prediction is not None:
                manual_state = manual_prediction
            chop_state = chop_prediction
            batch_raw_predictions.append(raw_prediction.detach().cpu())
            if manual_prediction is not None:
                batch_manual_predictions.append(manual_prediction.detach().cpu())
            batch_chop_predictions.append(chop_prediction.detach().cpu())
            batch_targets.append(target.detach().cpu())

        raw_rollouts.append(torch.stack(batch_raw_predictions, dim=1))
        if batch_manual_predictions:
            manual_rollouts.append(torch.stack(batch_manual_predictions, dim=1))
        chop_rollouts.append(torch.stack(batch_chop_predictions, dim=1))
        target_rollouts.append(torch.stack(batch_targets, dim=1))
        log(f"  completed {min(batch_start + current_batch_size, len(indices))}/{len(indices)} samples")

    dataset_output = output_dir / data_path.stem
    dataset_output.mkdir()
    horizon_summary = _summarize(rows, data_path.stem)
    overall_summary = _overall_summary(rows, data_path.stem)
    _write_csv(dataset_output / "rollout_per_sample_metrics.csv", rows)
    _write_csv(dataset_output / "rollout_horizon_summary.csv", horizon_summary)
    _write_csv(dataset_output / "rollout_method_summary.csv", overall_summary)
    if save_predictions:
        saved_rollouts = {
            "dataset": data_path.stem,
            "data_path": str(data_path.resolve()),
            "sample_ids": [f"sample_{int(index)}" for index in indices],
            "source_indices": torch.as_tensor(indices.copy(), dtype=torch.long),
            "seed": seed,
            "demo_num": DEMO_NUM,
            "rollout_steps": rollout_steps,
            "targets": torch.cat(target_rollouts, dim=0),
            "raw_icon": torch.cat(raw_rollouts, dim=0),
            "icon_chop": torch.cat(chop_rollouts, dim=0),
        }
        if manual_rollouts:
            saved_rollouts["manual_linear_prompt_scale"] = torch.cat(manual_rollouts, dim=0)
        torch.save(saved_rollouts, dataset_output / "rollouts.pt")
    return {
        "dataset": data_path.stem,
        "data_path": str(data_path.resolve()),
        "num_samples": len(indices),
        "sequence_shape": list(sequences.shape),
        "method_summary": overall_summary,
        "per_sample_csv": str((dataset_output / "rollout_per_sample_metrics.csv").resolve()),
        "horizon_summary_csv": str((dataset_output / "rollout_horizon_summary.csv").resolve()),
        "rollouts_pt": str((dataset_output / "rollouts.pt").resolve()) if save_predictions else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset filenames; default: every *.pt in data-dir")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--icon-chop-program", type=Path, default=DEFAULT_CHOP_PROGRAM)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logs-dir", type=Path, default=SCRIPT_DIR / "logs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--include-manual", action="store_true", help="Include the historical manual [-1,1] baseline")
    args = parser.parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.rollout_steps <= 0:
        parser.error("num-samples, batch-size, and rollout-steps must all be positive")
    return args


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    checkpoint = args.ckpt.resolve()
    chop_program = args.icon_chop_program.resolve()
    for required in (data_dir, checkpoint, chop_program):
        if not required.exists():
            raise FileNotFoundError(required)
    dataset_paths = (
        [data_dir / name for name in args.datasets]
        if args.datasets
        else sorted(data_dir.glob("*.pt"))
    )
    if not dataset_paths or any(not path.is_file() for path in dataset_paths):
        raise FileNotFoundError(f"Could not resolve all requested datasets under {data_dir}: {dataset_paths}")

    output_dir = _unique_output_dir(args.logs_dir.resolve(), args.run_name)
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    device = _resolve_device(args.device)
    log(f"output_dir={output_dir}")
    log(f"device={device} seed={args.seed} batch_size={args.batch_size}")
    log(f"checkpoint={checkpoint}")
    log(f"icon_chop_program={chop_program}")
    model = load_icon_model(checkpoint, device)
    wrapper = IconWrapper(model)
    chop_runner = _load_chop_runner(chop_program)
    dataset_summaries = []
    for data_path in dataset_paths:
        dataset_summaries.append(
            evaluate_dataset(
                data_path=data_path.resolve(),
                wrapper=wrapper,
                chop_runner=chop_runner,
                device=device,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                rollout_steps=args.rollout_steps,
                seed=args.seed,
                output_dir=output_dir,
                save_predictions=not args.no_save_predictions,
                include_manual=args.include_manual,
                log=log,
            )
        )
    summary = {
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(checkpoint),
        "icon_chop_program": str(chop_program),
        "data_dir": str(data_dir),
        "datasets": dataset_summaries,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "seed_evidence": "Historical rollout starts sample_90,sample_254,sample_283,..., matching RandomState(0).permutation(500).",
        "batch_size": args.batch_size,
        "demo_num": DEMO_NUM,
        "rollout_steps": args.rollout_steps,
        "device": str(device),
        "save_predictions": not args.no_save_predictions,
        "include_manual": args.include_manual,
    }
    summary_path = output_dir / "summary.json"
    summary["summary_json"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
