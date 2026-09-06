from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _public_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only aggregate, non-sample-specific diagnostics in evaluator output."""
    public: dict[str, Any] = {}
    for key in (
        "num_batches",
        "num_samples",
        "device",
        "pollutants",
        "context_strategy",
        "retrieval_index",
        "similar_len",
        "graph",
    ):
        if key in diagnostics:
            public[key] = diagnostics[key]
    return public


def format_summary(metrics: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> str:
    graph = diagnostics.get("graph") or {}
    pollutants = ",".join(str(name) for name in diagnostics.get("pollutants", [])) or "pollutants"
    return (
        f"score={float(metrics['score']):.6f}; "
        f"baseline_rel_l2={float(metrics['baseline_rel_l2']):.6f}; "
        f"chain_rel_l2={float(metrics['chain_rel_l2']):.6f}; "
        f"pollutants={pollutants}; "
        f"baseline_pollutant_rmse={float(metrics['baseline_pollutant_rmse']):.6f}; "
        f"chain_pollutant_rmse={float(metrics['chain_pollutant_rmse']):.6f}; "
        f"success_rate={float(metrics['success_rate']):.2%}; "
        f"edges={int(graph.get('edges', 0))}"
    )


def build_score_payload(
    *,
    result: Mapping[str, Any],
    program_path: str | Path,
    ckpt_path: str | Path,
    data_dir: str | Path,
    dataset_name: str,
) -> dict[str, Any]:
    metrics = dict(result["metrics"])
    diagnostics = _public_diagnostics(dict(result.get("diagnostics") or {}))
    payload = {
        "score": float(metrics["score"]),
        "summary": format_summary(metrics, diagnostics),
        "program_path": str(Path(program_path).resolve()),
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_name": dataset_name,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    payload["feedback"] = "\n".join(
        [
            payload["summary"],
            (
                f"baseline_norm_mse={float(metrics['baseline_norm_mse']):.8f} | "
                f"chain_norm_mse={float(metrics['chain_norm_mse']):.8f}"
            ),
            *[
                f"baseline_{name}_rmse={float(metrics[f'baseline_{name}_rmse']):.6f} | "
                f"chain_{name}_rmse={float(metrics[f'chain_{name}_rmse']):.6f}"
                for name in diagnostics.get("pollutants", [])
                if f"baseline_{name}_rmse" in metrics and f"chain_{name}_rmse" in metrics
            ],
        ]
    )
    return payload


def build_failure_payload(summary: str, *, detail: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "score": 0.0,
        "summary": summary,
        "success_rate": 0.0,
        "failure_count": 1,
    }
    if detail:
        payload["error"] = detail
    return payload
