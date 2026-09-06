from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import direct_year
import retrieval


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
GICON_CODE_ROOT = WORKSPACE_ROOT / "graph_in_context" / "gicon_chop_code"
DEFAULT_PROGRAM = GICON_CODE_ROOT / "seed" / "gicon_chain"
DEFAULT_DTS = (1, 12, 24, 36, 48, 72)
DEFAULT_STRATEGIES = ("random", "retrieval")
REGIONS = ("bthsa", "yrd")


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    # Preserve the spelling of a user-supplied path instead of resolving links.
    return Path(os.path.abspath(path))


def _infer_model_region(ckpt_path: Path) -> str:
    parts = [part.lower() for part in ckpt_path.parts]
    matches = [region for region in REGIONS if region in parts]
    if len(matches) == 1:
        return matches[0]
    normalized = str(ckpt_path).lower().replace("-", "_")
    matches = [region for region in REGIONS if region in normalized]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        "Could not infer the checkpoint region from its path. Pass --model-region bthsa or --model-region yrd."
    )


def _resolve_regions(model_region: str, eval_region: str, ckpt_path: Path) -> tuple[str, str]:
    model = _infer_model_region(ckpt_path) if model_region == "auto" else model_region
    evaluation = ("yrd" if model == "bthsa" else "bthsa") if eval_region == "auto" else eval_region
    return model, evaluation


def parse_args() -> argparse.Namespace:
    default_output = SCRIPT_DIR / "logs" / datetime.now().strftime("gicon_%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the saved full-year GICON-CHOP cross-region sweep. By default this runs "
            "all six forecast horizons with random and retrieval contexts."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing dataset_*.nc, stations_*.csv, altitude.npy.")
    parser.add_argument("--ckpt", required=True, help="One regional GICON checkpoint.")
    parser.add_argument("--program", default=str(DEFAULT_PROGRAM))
    parser.add_argument("--model-region", choices=("auto", *REGIONS), default="auto")
    parser.add_argument("--eval-region", choices=("auto", *REGIONS), default="auto")
    parser.add_argument("--station-file", default=None)
    parser.add_argument("--altitude-file", default=None)
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--deltas", nargs="+", type=int, default=list(DEFAULT_DTS))
    parser.add_argument("--strategies", nargs="+", choices=DEFAULT_STRATEGIES, default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--query-start", default="2023-01-01T00:00:00")
    parser.add_argument("--query-end", default="2023-12-31T23:00:00")
    parser.add_argument("--context-start", default="2016-01-01T00:00:00")
    parser.add_argument("--context-end", default="2022-12-31T23:00:00")
    parser.add_argument("--data-start", default="2016-01-01T00:00:00")
    parser.add_argument("--data-end", default="2023-12-31T23:00:00")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-num", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--similar-len", type=int, default=24)
    parser.add_argument("--device", default="cpu", help="Historical saved results used CPU.")
    parser.add_argument("--no-altitude", action="store_true")
    parser.add_argument(
        "--retrieval-index-dir",
        default=str(SCRIPT_DIR / "cache" / "retrieval_indices"),
        help="Directory containing or receiving cached retrieval indices.",
    )
    parser.add_argument("--retrieval-backend", choices=("auto", "faiss", "numpy"), default="auto")
    parser.add_argument("--retrieval-query-chunk-size", type=int, default=128)
    parser.add_argument(
        "--build-missing-retrieval-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the deterministic local retrieval index when it is absent.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute leaves that already contain score.yaml.")
    return parser.parse_args()


def _normalize_data_dir(data_dir: Path) -> Path:
    if any(data_dir.glob("dataset_*.nc")):
        return data_dir
    nested = data_dir / "eval_air"
    if any(nested.glob("dataset_*.nc")):
        return nested.resolve()
    return data_dir


def _retrieval_pattern(
    eval_region: str,
    query_start: str,
    delta_t: int,
    demo_num: int,
    window_size: int,
    similar_len: int,
) -> str:
    return (
        f"{eval_region}-{query_start[:4]}_multiframe_dt{delta_t}_demo{demo_num}_"
        f"win{window_size}_sim{similar_len}*.npz"
    )


def _retrieval_index_path(
    *,
    retrieval_module: Any,
    index_dir: Path,
    data_dir: Path,
    eval_region: str,
    query_start: str,
    query_end: str,
    context_start: str,
    context_end: str,
    data_start: str,
    data_end: str,
    delta_t: int,
    demo_num: int,
    window_size: int,
    similar_len: int,
    seed: int,
    backend: str,
    query_chunk_size: int,
    build_missing: bool,
) -> Path:
    pattern = _retrieval_pattern(eval_region, query_start, delta_t, demo_num, window_size, similar_len)
    matches = sorted(index_dir.glob(pattern))
    if matches:
        return matches[-1].resolve()
    if not build_missing:
        raise FileNotFoundError(f"No retrieval index matched {index_dir / pattern}")

    output_path = index_dir / pattern.replace("*.npz", ".npz")
    config = retrieval_module.RetrievalBuildConfig(
        output_path=output_path,
        data_dir=data_dir,
        eval_region=eval_region,
        query_start=query_start,
        query_end=query_end,
        context_start=context_start,
        context_end=context_end,
        data_start=data_start,
        data_end=data_end,
        delta_t=delta_t,
        demo_num=demo_num,
        window_size=window_size,
        similar_len=similar_len,
        seed=seed,
        backend=backend,
    )
    print(f"[index] building {output_path.name}; this is reused by later runs", flush=True)
    retrieval_module.build_retrieval_index(config, query_chunk_size=query_chunk_size)
    return output_path.resolve()


def _sample_stat(payload: dict[str, Any], metric: str, stat: str) -> str:
    value = (
        payload.get("per_sample_metric_summary", {})
        .get("metrics", {})
        .get(metric, {})
        .get(stat)
    )
    return f"{value:.9g}" if isinstance(value, (int, float)) else ""


def _summary_row(pairing: str, delta_t: int, strategy: str, score: dict[str, Any], status: str) -> dict[str, Any]:
    def number(key: str) -> str:
        value = score.get(key)
        return f"{value:.9g}" if isinstance(value, (int, float)) else ""

    return {
        "pairing": pairing,
        "delta_t": delta_t,
        "strategy": strategy,
        "score": number("score"),
        "sample_score_mean": _sample_stat(score, "score", "mean"),
        "sample_score_std": _sample_stat(score, "score", "std"),
        "pm25_score": number("pm25_score"),
        "sample_pm25_score_mean": _sample_stat(score, "pm25_score", "mean"),
        "sample_pm25_score_std": _sample_stat(score, "pm25_score", "std"),
        "o3_score": number("o3_score"),
        "sample_o3_score_mean": _sample_stat(score, "o3_score", "mean"),
        "sample_o3_score_std": _sample_stat(score, "o3_score", "std"),
        "status": status,
    }


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: (row["pairing"], int(row["delta_t"]), row["strategy"]))
    fields = list(_summary_row("", 0, "", {}, "").keys())
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# GICON-CHOP full-year evaluation",
        "",
        "2023 hourly direct-year protocol with 2016-2022 contexts.",
        "",
        "| pairing | delta_t | context | score | PM2.5 | O3 | status |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['pairing']} | {row['delta_t']} | {row['strategy']} | {row['score']} | "
            f"{row['pm25_score']} | {row['o3_score']} | {row['status']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    args = parse_args()

    input_data_dir = _normalize_data_dir(_path(args.data_dir))
    data_dir = input_data_dir
    ckpt_path = _path(args.ckpt)
    program_path = _path(args.program)
    output_dir = _path(args.output_dir)
    index_dir = _path(args.retrieval_index_dir)
    model_region, eval_region = _resolve_regions(args.model_region, args.eval_region, ckpt_path)
    station_file = _path(args.station_file) if args.station_file else data_dir / f"stations_{eval_region}.csv"
    altitude_file = _path(args.altitude_file) if args.altitude_file else data_dir / "altitude.npy"
    pairing = f"{model_region}_model_{eval_region}_data"

    required = {
        "checkpoint": ckpt_path,
        "program": program_path,
        "evaluation NetCDF": data_dir / f"dataset_{eval_region}.nc",
        "station file": station_file,
        "altitude file": altitude_file,
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if any(delta <= 0 for delta in args.deltas):
        raise ValueError("All --deltas must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "protocol": "gicon_2023_full_year_direct",
        "pairing": pairing,
        "model_region": model_region,
        "eval_region": eval_region,
        "program": str(program_path),
        "ckpt": str(ckpt_path),
        "input_data_dir": str(input_data_dir),
        "data_dir": str(data_dir),
        "station_file": str(station_file),
        "altitude_file": str(altitude_file),
        "deltas": args.deltas,
        "strategies": args.strategies,
        "query_start": args.query_start,
        "query_end": args.query_end,
        "context_start": args.context_start,
        "context_end": args.context_end,
        "data_start": args.data_start,
        "data_end": args.data_end,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "demo_num": args.demo_num,
        "window_size": args.window_size,
        "similar_len": args.similar_len,
        "device": args.device,
        "use_altitude": not args.no_altitude,
        "retrieval_index_dir": str(index_dir),
        "retrieval_backend": args.retrieval_backend,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"GICON protocol: {pairing}; deltas={args.deltas}; strategies={args.strategies}")
    print(f"Device: {args.device}; checkpoint: {ckpt_path}")
    rows: list[dict[str, Any]] = []
    failures = 0

    for delta_t in args.deltas:
        for strategy in args.strategies:
            tag = f"{pairing} dt{delta_t} {strategy}"
            leaf = output_dir / pairing / f"dt{delta_t}" / strategy
            raw_path = leaf / "raw_evaluator.json"
            score_path = leaf / "score.yaml"
            leaf.mkdir(parents=True, exist_ok=True)
            if score_path.exists() and raw_path.exists() and not args.force:
                print(f"[skip] {tag}: output already exists", flush=True)
                score = _load_yaml(score_path)
            else:
                print(f"[run ] {tag}", flush=True)
                try:
                    retrieval_path = None
                    if strategy == "retrieval":
                        retrieval_path = _retrieval_index_path(
                            retrieval_module=retrieval,
                            index_dir=index_dir,
                            data_dir=data_dir,
                            eval_region=eval_region,
                            query_start=args.query_start,
                            query_end=args.query_end,
                            context_start=args.context_start,
                            context_end=args.context_end,
                            data_start=args.data_start,
                            data_end=args.data_end,
                            delta_t=delta_t,
                            demo_num=args.demo_num,
                            window_size=args.window_size,
                            similar_len=args.similar_len,
                            seed=args.seed,
                            backend=args.retrieval_backend,
                            query_chunk_size=args.retrieval_query_chunk_size,
                            build_missing=args.build_missing_retrieval_index,
                        )
                    config = direct_year.DirectYearConfig(
                        program_path=program_path,
                        data_dir=data_dir,
                        ckpt_path=ckpt_path,
                        station_file=station_file,
                        altitude_file=altitude_file,
                        model_region=model_region,
                        eval_region=eval_region,
                        query_start=args.query_start,
                        query_end=args.query_end,
                        context_start=args.context_start,
                        context_end=args.context_end,
                        data_start=args.data_start,
                        data_end=args.data_end,
                        delta_t=delta_t,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        demo_num=args.demo_num,
                        window_size=args.window_size,
                        context_strategy=strategy,
                        retrieval_index_path=retrieval_path,
                        similar_len=args.similar_len,
                        device_name=args.device,
                        use_altitude=not args.no_altitude,
                    )
                    payload = direct_year.evaluate_direct_year(config)
                    payload["output_json"] = str(raw_path)
                    direct_year.write_direct_outputs(payload, raw_path, score_path)
                    error_path = leaf / "error.txt"
                    if error_path.exists():
                        error_path.unlink()
                    score = _load_yaml(score_path)
                    print(
                        f"[done] {tag}: score={score.get('score')} "
                        f"pm25={score.get('pm25_score')} o3={score.get('o3_score')}",
                        flush=True,
                    )
                except Exception:
                    failures += 1
                    (leaf / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    print(f"[FAIL] {tag}: see {leaf / 'error.txt'}", file=sys.stderr, flush=True)
                    score = {}

            status = "ok" if score else "error"
            rows.append(_summary_row(pairing, delta_t, strategy, score, status))
            _write_summary(output_dir, rows)

    print(f"Sweep finished: {len(rows)} leaves, {failures} runtime failures. Summary: {output_dir / 'summary.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
