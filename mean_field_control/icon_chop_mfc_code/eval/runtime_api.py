from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def resolve_data_sources(data_dir: str | Path, data_glob: str) -> list[str]:
    resolved_dir = Path(data_dir).resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {resolved_dir}")

    data_sources = sorted(str(path) for path in resolved_dir.glob(data_glob))
    if not data_sources:
        raise FileNotFoundError(f"No evaluation data matched {resolved_dir / data_glob}")
    return data_sources


def build_score_payload(
    *,
    result: Mapping[str, Any],
    program_path: str | Path,
    ckpt_path: str | Path,
    data_dir: str | Path,
    data_glob: str,
) -> dict[str, Any]:
    metrics = dict(result["metrics"])
    diagnostics = dict(result.get("diagnostics") or {})
    score = float(metrics["score"])
    payload = {
        "score": score,
        "summary": format_summary(metrics, diagnostics),
        "program_path": str(Path(program_path).resolve()),
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "data_glob": str(data_glob),
        "baseline_rel_l2": float(metrics["baseline_rel_l2"]),
        "chain_rel_l2": float(metrics["chain_rel_l2"]),
        "mean_relative_improve": float(metrics["mean_relative_improve"]),
        "var_rel_improve": float(metrics["var_rel_improve"]),
        "var_chain_rel_l2": float(metrics["var_chain_rel_l2"]),
        "improve_ratio": float(metrics["improve_ratio"]),
        "regress_ratio": float(metrics["regress_ratio"]),
        "success_rate": float(metrics["success_rate"]),
        "failure_count": int(metrics["failure_count"]),
        "baseline_mse": float(metrics["baseline_mse"]),
        "chain_mse": float(metrics["chain_mse"]),
        "num_datasets": int(diagnostics.get("num_datasets", 0)),
        "dataset_summaries": diagnostics.get("dataset_summaries") or [],
        "best_sample": diagnostics.get("best_sample"),
        "worst_sample": diagnostics.get("worst_sample"),
    }
    payload["feedback"] = "\n".join(format_feedback_lines(payload))
    return payload


def build_failure_payload(
    summary: str,
    *,
    detail: str | None = None,
    program_path: str | Path | None = None,
    ckpt_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    data_glob: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": 0.0,
        "summary": summary,
        "success_rate": 0.0,
        "failure_count": 1,
    }
    if detail:
        payload["error"] = detail
    if program_path is not None:
        payload["program_path"] = str(Path(program_path).resolve())
    if ckpt_path is not None:
        payload["checkpoint_path"] = str(Path(ckpt_path).resolve())
    if data_dir is not None:
        payload["data_dir"] = str(Path(data_dir).resolve())
    if data_glob is not None:
        payload["data_glob"] = str(data_glob)
    return payload


def format_summary(metrics: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    score = float(metrics["score"])
    baseline_rel_l2 = float(metrics["baseline_rel_l2"])
    chain_rel_l2 = float(metrics["chain_rel_l2"])
    success_rate = float(metrics["success_rate"])
    failure_count = int(metrics["failure_count"])
    num_datasets = int(diagnostics.get("num_datasets", 0))

    summary = (
        f"score={score:.6f}; "
        f"baseline_rel_l2={baseline_rel_l2:.6f}; "
        f"chain_rel_l2={chain_rel_l2:.6f}; "
        f"success_rate={success_rate:.2%}; "
        f"failures={failure_count}; "
        f"datasets={num_datasets}"
    )
    if score > 1.0:
        summary += "; beats_baseline=true"
    return summary


def format_feedback_lines(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        (
            f"datasets={int(payload.get('num_datasets', 0))} | "
            f"success_rate={float(payload.get('success_rate', 0.0)):.2%} | "
            f"failures={int(payload.get('failure_count', 0))}"
        ),
        (
            f"score={float(payload.get('score', 0.0)):.6f} | "
            f"baseline_rel_l2={float(payload.get('baseline_rel_l2', 0.0)):.6f} | "
            f"chain_rel_l2={float(payload.get('chain_rel_l2', 0.0)):.6f}"
        ),
    ]

    if "mean_relative_improve" in payload:
        lines.append(
            (
                f"mean_relative_improve={float(payload.get('mean_relative_improve', 0.0)):.6f} | "
                f"var_rel_improve={float(payload.get('var_rel_improve', 0.0)):.6f} | "
                f"var_chain_rel_l2={float(payload.get('var_chain_rel_l2', 0.0)):.6f}"
            )
        )
    if "improve_ratio" in payload:
        lines.append(
            (
                f"improve_ratio={float(payload.get('improve_ratio', 0.0)):.2%} | "
                f"regress_ratio={float(payload.get('regress_ratio', 0.0)):.2%}"
            )
        )
    if "baseline_mse" in payload:
        lines.append(
            (
                f"baseline_mse={float(payload.get('baseline_mse', 0.0)):.6f} | "
                f"chain_mse={float(payload.get('chain_mse', 0.0)):.6f}"
            )
        )

    best_sample = payload.get("best_sample")
    if isinstance(best_sample, Mapping):
        lines.append(
            "best_sample="
            + repr(
                {
                    "dataset_name": best_sample.get("dataset_name"),
                    "sample_id": best_sample.get("sample_id"),
                    "improvement_ratio": best_sample.get("improvement_ratio"),
                }
            )
        )

    worst_sample = payload.get("worst_sample")
    if isinstance(worst_sample, Mapping):
        lines.append(
            "worst_sample="
            + repr(
                {
                    "dataset_name": worst_sample.get("dataset_name"),
                    "sample_id": worst_sample.get("sample_id"),
                    "improvement_ratio": worst_sample.get("improvement_ratio"),
                }
            )
        )
    return lines
