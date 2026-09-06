from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


def _testbed_root() -> Path:
    return Path(__file__).resolve().parent


def _bootstrap_paths() -> None:
    for path in (_testbed_root(),):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _strip_env_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid env line in {path}:{line_number}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty env key in {path}:{line_number}")
        os.environ.setdefault(key, _strip_env_quotes(value))


def _env_file_from_argv() -> tuple[str, bool]:
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--env-file" and index + 1 < len(args):
            return args[index + 1], True
        if arg.startswith("--env-file="):
            return arg.split("=", 1)[1], True
    return "testbed.env", False


def _load_env_defaults() -> str:
    env_file, explicit = _env_file_from_argv()
    if not env_file:
        return ""
    path = Path(env_file)
    if not path.is_absolute():
        path = _testbed_root() / path
    if explicit and not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    _read_env_file(path)
    return env_file


def _default_from_env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    return fallback


def parse_args() -> argparse.Namespace:
    env_file = _load_env_defaults()
    parser = argparse.ArgumentParser(description="Run a standalone GICON operator-chain testbed.")
    parser.add_argument("--env-file", default=env_file)
    parser.add_argument("--program", default="seed/gicon_chain")
    parser.add_argument("--model-region", default=_default_from_env("GICON_TESTBED_MODEL_REGION", "bthsa"))
    parser.add_argument(
        "--eval-region",
        "--dataset-name",
        dest="eval_region",
        default=_default_from_env("GICON_TESTBED_EVAL_REGION", _default_from_env("GICON_TESTBED_DATASET", "yrd")),
    )
    parser.add_argument("--data-dir", default=_default_from_env("GICON_TESTBED_DATA_DIR", "data"))
    parser.add_argument("--ckpt", default=_default_from_env("GICON_TESTBED_CKPT"))
    parser.add_argument("--station-file", default=_default_from_env("GICON_TESTBED_STATION_FILE"))
    parser.add_argument("--altitude-file", default=_default_from_env("GICON_TESTBED_ALTITUDE_FILE"))
    parser.add_argument("--no-altitude", action="store_true")
    parser.add_argument("--device", default=_default_from_env("GICON_TESTBED_DEVICE"))
    parser.add_argument("--batch-size", type=int, default=int(_default_from_env("GICON_TESTBED_BATCH_SIZE", "1")))
    parser.add_argument("--num-samples", type=int, default=int(_default_from_env("GICON_TESTBED_NUM_SAMPLES", "1")))
    parser.add_argument("--seed", type=int, default=int(_default_from_env("GICON_TESTBED_SEED", "42")))
    parser.add_argument("--query-sample-mode", default=_default_from_env("GICON_TESTBED_QUERY_SAMPLE_MODE", "random"))
    parser.add_argument("--demo-num", type=int, default=int(_default_from_env("GICON_TESTBED_DEMO_NUM", "5")))
    parser.add_argument("--delta-t", type=int, default=int(_default_from_env("GICON_TESTBED_DELTA_T", "24")))
    parser.add_argument("--window-size", type=int, default=int(_default_from_env("GICON_TESTBED_WINDOW_SIZE", "24")))
    parser.add_argument("--query-start", default=_default_from_env("GICON_TESTBED_QUERY_START", "2023-01-01T00:00:00"))
    parser.add_argument("--query-end", default=_default_from_env("GICON_TESTBED_QUERY_END", "2023-01-03T00:00:00"))
    parser.add_argument("--context-start", default=_default_from_env("GICON_TESTBED_CONTEXT_START", "2016-01-01T00:00:00"))
    parser.add_argument("--context-end", default=_default_from_env("GICON_TESTBED_CONTEXT_END", "2022-12-31T00:00:00"))
    parser.add_argument("--data-start", default=_default_from_env("GICON_TESTBED_DATA_START", "2016-01-01T00:00:00"))
    parser.add_argument("--data-end", default=_default_from_env("GICON_TESTBED_DATA_END", "2023-12-31T00:00:00"))
    parser.add_argument(
        "--context-strategy",
        choices=("random", "retrieval"),
        default=_default_from_env("GICON_TESTBED_CONTEXT_STRATEGY", "random"),
    )
    parser.add_argument("--retrieval-index", default=_default_from_env("GICON_TESTBED_RETRIEVAL_INDEX"))
    parser.add_argument(
        "--similar-len",
        type=int,
        default=int(
            _default_from_env(
                "GICON_TESTBED_SIMILAR_LEN",
                _default_from_env("GICON_TESTBED_WINDOW_SIZE", "24"),
            )
        ),
    )
    parser.add_argument("--output-json", default=_default_from_env("GICON_TESTBED_OUTPUT_JSON"))
    return parser.parse_args()


def _resolve_path(base_root: Path, value: str | None, fallback: Path | None = None) -> Path:
    path = Path(value) if value else fallback
    if path is None:
        raise ValueError("Missing path")
    if not path.is_absolute():
        path = base_root / path
    return path.resolve()


def _resolve_program_path(testbed_root: Path, program: str) -> Path:
    path = Path(program)
    if not path.is_absolute():
        path = testbed_root / path
    return path.resolve()


def _resolve_output_path(testbed_root: Path, output_json: str | None) -> Path:
    if output_json:
        return _resolve_path(testbed_root, output_json)
    return (testbed_root / "results" / "summary.json").resolve()


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run() -> dict[str, object]:
    _bootstrap_paths()

    from runtime.data import GiconDataConfig, default_altitude_path, default_checkpoint_path, default_station_path
    from runtime.eval_core import evaluate_program_file
    from runtime.runtime_api import build_score_payload

    args = parse_args()
    testbed_root = _testbed_root()
    model_region = str(args.model_region).lower()
    eval_region = str(args.eval_region).lower()
    data_dir = _resolve_path(testbed_root, args.data_dir)
    ckpt_path = _resolve_path(testbed_root, args.ckpt, default_checkpoint_path(model_region))
    station_file = _resolve_path(testbed_root, args.station_file, default_station_path(data_dir, eval_region))
    altitude_file = _resolve_path(testbed_root, args.altitude_file, default_altitude_path(data_dir))
    program_path = _resolve_program_path(testbed_root, args.program)
    retrieval_index_path = (
        _resolve_path(testbed_root, args.retrieval_index)
        if args.retrieval_index
        else None
    )

    config = GiconDataConfig(
        data_dir=data_dir,
        dataset_name=eval_region,
        query_start=args.query_start,
        query_end=args.query_end,
        context_start=args.context_start,
        context_end=args.context_end,
        data_start=args.data_start,
        data_end=args.data_end,
        delta_t=args.delta_t,
        window_size=args.window_size,
        demo_num=args.demo_num,
        query_sample_mode=args.query_sample_mode,
        seed=args.seed,
        context_strategy=args.context_strategy,
        retrieval_index_path=retrieval_index_path,
        similar_len=args.similar_len,
    )
    result = evaluate_program_file(
        program_path=program_path,
        ckpt_path=ckpt_path,
        data_config=config,
        station_file=station_file,
        altitude_file=altitude_file,
        use_altitude=not args.no_altitude,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device_name=args.device,
    )
    payload = build_score_payload(
        result=result,
        program_path=program_path,
        ckpt_path=ckpt_path,
        data_dir=data_dir,
        dataset_name=eval_region,
    )
    payload.update(
        {
            "model_region": model_region,
            "eval_region": eval_region,
            "env_file": args.env_file,
            "pollutants": ["pm25", "o3"],
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "query_sample_mode": args.query_sample_mode,
            "context_strategy": args.context_strategy,
            "retrieval_index": str(retrieval_index_path) if retrieval_index_path is not None else None,
            "similar_len": args.similar_len,
            "demo_num": args.demo_num,
            "delta_t": args.delta_t,
            "window_size": args.window_size,
            "station_file": str(station_file),
            "altitude_file": str(altitude_file),
            "use_altitude": not args.no_altitude,
        }
    )
    return payload


def main() -> int:
    _bootstrap_paths()
    args = parse_args()
    output_path = _resolve_output_path(_testbed_root(), args.output_json)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            payload = run()
    except Exception as exc:  # noqa: BLE001
        from runtime.runtime_api import build_failure_payload

        payload = build_failure_payload("gicon testbed evaluation failed", detail=str(exc))
    payload["output_json"] = str(output_path)
    _write_payload(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
