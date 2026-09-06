from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _add_repo_paths(repo_root: Path) -> None:
    eval_dir = repo_root / "eval"
    seed_dir = repo_root / "seed"
    for path in (eval_dir, seed_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _default_from_env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None and value != "":
        return value
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the ICON chain and emit a single JSON payload for Escher Loop."
    )
    parser.add_argument("--program", default="seed/initial_program.py")
    parser.add_argument("--ckpt", default=_default_from_env("ICON_CHOP_CKPT_PATH"))
    parser.add_argument("--data-dir", default=_default_from_env("ICON_CHOP_DATA_DIR"))
    parser.add_argument(
        "--data-glob",
        default=_default_from_env("ICON_CHOP_DATA_GLOB", "*.h5"),
    )
    parser.add_argument(
        "--device", default=_default_from_env("ICON_CHOP_DEVICE")
    )
    parser.add_argument("--batch-size", type=int, default=int(_default_from_env("ICON_CHOP_BATCH_SIZE", "16")))
    parser.add_argument("--num-samples", type=int, default=int(_default_from_env("ICON_CHOP_NUM_SAMPLES", "256")))
    parser.add_argument("--seed", type=int, default=int(_default_from_env("ICON_CHOP_SEED", "0")))
    parser.add_argument("--demo-num", type=int, default=int(_default_from_env("ICON_CHOP_DEMO_NUM", "5")))
    parser.add_argument("--max-len", type=int, default=int(_default_from_env("ICON_CHOP_MAX_LEN", "100")))
    parser.add_argument("--k-dim", type=int, default=int(_default_from_env("ICON_CHOP_K_DIM", "2")))
    parser.add_argument(
        "--model-in-features",
        type=int,
        default=int(_default_from_env("ICON_CHOP_MODEL_IN_FEATURES", "3")),
    )
    parser.add_argument(
        "--output-json",
        default=_default_from_env("ICON_CHOP_OUTPUT_JSON"),
        help="Optional path for the full JSON payload. Defaults to results/escher_eval/summary.json.",
    )
    return parser.parse_args()


def _resolve_program_path(repo_root: Path, program_arg: str) -> Path:
    program_path = Path(program_arg)
    if not program_path.is_absolute():
        program_path = repo_root / program_path
    return program_path.resolve()


def _resolve_output_path(repo_root: Path, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        return output_path.resolve()
    return (repo_root / "results" / "escher_eval" / "summary.json").resolve()


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run() -> dict[str, object]:
    repo_root = _resolve_repo_root()
    _add_repo_paths(repo_root)
    args = parse_args()

    from eval_core import evaluate_program_file
    from runtime_api import build_score_payload, resolve_data_sources

    program_path = _resolve_program_path(repo_root, args.program)
    if not program_path.exists():
        raise FileNotFoundError(f"Program file not found: {program_path}")

    if args.ckpt is None:
        raise ValueError("Missing checkpoint path. Pass --ckpt or set ICON_CHOP_CKPT_PATH.")
    if args.data_dir is None:
        raise ValueError("Missing data directory. Pass --data-dir or set ICON_CHOP_DATA_DIR.")

    ckpt_path = Path(args.ckpt).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_sources = resolve_data_sources(data_dir, args.data_glob)
    result = evaluate_program_file(
        program_path=str(program_path),
        ckpt_path=str(ckpt_path),
        data_sources=data_sources,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        demo_num=args.demo_num,
        max_len=args.max_len,
        k_dim=args.k_dim,
        model_in_features=args.model_in_features,
        device_name=args.device,
    )
    payload = build_score_payload(
        result=result,
        program_path=program_path,
        ckpt_path=ckpt_path,
        data_dir=data_dir,
        data_glob=args.data_glob,
    )
    payload["batch_size"] = args.batch_size
    payload["num_samples"] = args.num_samples
    payload["seed"] = args.seed
    payload["demo_num"] = args.demo_num
    payload["max_len"] = args.max_len
    payload["k_dim"] = args.k_dim
    payload["model_in_features"] = args.model_in_features
    if args.device:
        payload["device"] = args.device

    output_path = _resolve_output_path(repo_root, args.output_json)
    payload["output_json"] = str(output_path)
    _write_payload(output_path, payload)
    return payload


def main() -> int:
    repo_root = _resolve_repo_root()
    _add_repo_paths(repo_root)
    args = parse_args()
    output_path = _resolve_output_path(repo_root, args.output_json)

    try:
        payload = run()
    except Exception as exc:  # noqa: BLE001
        from runtime_api import build_failure_payload

        payload = build_failure_payload(
            "icon chop evaluation failed",
            detail=str(exc),
            program_path=_resolve_program_path(repo_root, args.program),
            ckpt_path=args.ckpt or "<missing>",
            data_dir=args.data_dir or "<missing>",
            data_glob=args.data_glob,
        )
        payload["output_json"] = str(output_path)
        payload["batch_size"] = args.batch_size
        payload["num_samples"] = args.num_samples
        payload["seed"] = args.seed
        payload["demo_num"] = args.demo_num
        payload["max_len"] = args.max_len
        payload["k_dim"] = args.k_dim
        payload["model_in_features"] = args.model_in_features
        if args.device:
            payload["device"] = args.device
        _write_payload(output_path, payload)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
