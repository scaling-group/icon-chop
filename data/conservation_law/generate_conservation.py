"""Self-contained generator for the verified conservation-law datasets.

The module only imports files shipped in ``conservation_law/``.  Historical
Python-random window selections are supplied by ``window_starts.py``; all
remaining randomness is reconstructed from the recorded Torch/Python seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import struct
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from einops import repeat
from omegaconf import OmegaConf

import data_utils
from window_starts import EVAL_STARTS, EVOLVE_STARTS
from weno_new import weno_roll, weno_scheme


FLUX_NAMES = ("sin", "tanh", "u2")
LENGTH = 100
EQNS = 5
NUM = 100
TRAJECTORIES = EQNS * NUM
DT = 0.0005
DX = 0.01
STRIDE = 200

EVAL_SEED = 1234567890123
EVOLVE_SEED = 123456789

EVAL_SAVED_START_HASHES = {
    "sin": "a15772406490b116bf80506733bbf052df0641d5c15025aca06e3759c6a69088",
    "tanh": "fc625934bc2a2ba5bb28e0f64b77371769181dadb23efca95dd99b29670e3cfe",
    "u2": "5d7be2142962f6177a6a10bba2610f012aebaa0743addf778cd4aa4e860b0379",
}
EVOLVE_RAW_START_HASH = "68a74944dfd3e3baf4cd74db577b803bc31acd1ebdd65cf511f31b4abd7ceddc"
EVOLVE_SAVED_START_HASH = "f2090a93198b111ecb0dd26b107c1b4c4bc2577d2f7501941e1fa2713c63c17d"


def int64_sha256(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def permutation(seed: int, count: int) -> torch.Tensor:
    # Historical generation used CUDA for every stochastic tensor before this
    # CPU randperm, so a dedicated CPU generator reproduces its exact state.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(count, generator=generator)


def validate_recovered_starts() -> None:
    eval_perm = permutation(EVAL_SEED, TRAJECTORIES)
    for flux_name in FLUX_NAMES:
        starts = EVAL_STARTS[flux_name]
        if len(starts) != TRAJECTORIES or not all(0 <= x <= 200 for x in starts):
            raise RuntimeError(f"Invalid recovered eval starts for {flux_name}")
        saved_starts = torch.tensor(starts, dtype=torch.int64)[eval_perm].tolist()
        if int64_sha256(saved_starts) != EVAL_SAVED_START_HASHES[flux_name]:
            raise RuntimeError(f"Recovered eval start hash failed for {flux_name}")

    if len(EVOLVE_STARTS) != 2 * TRAJECTORIES:
        raise RuntimeError("Invalid recovered evolve start count")
    if not all(0 <= x <= 300 for x in EVOLVE_STARTS):
        raise RuntimeError("Invalid recovered evolve start range")
    if int64_sha256(EVOLVE_STARTS) != EVOLVE_RAW_START_HASH:
        raise RuntimeError("Recovered evolve raw start hash failed")
    evolve_perm = permutation(EVOLVE_SEED, 2 * TRAJECTORIES)
    saved = torch.tensor(EVOLVE_STARTS, dtype=torch.int64)[evolve_perm].tolist()
    if int64_sha256(saved) != EVOLVE_SAVED_START_HASH:
        raise RuntimeError("Recovered evolve saved start hash failed")


def flux_functions(
    name: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], str]:
    if name == "sin":
        return (
            lambda u: torch.sin(u) - torch.cos(u),
            lambda u: torch.cos(u) + torch.sin(u),
            "sin(u) - cos(u)",
        )
    if name == "tanh":
        return (
            lambda u: torch.tanh(u),
            lambda u: 1.0 - torch.tanh(u) ** 2,
            "tanh(u)",
        )
    if name == "u2":
        return (
            lambda u: u**2 / (u**2 + (1.0 - u) ** 2),
            lambda u: 2.0 * u * (1.0 - u) / (u**2 + (1.0 - u) ** 2) ** 2,
            "u^2 / (u^2 + (1-u)^2)",
        )
    raise ValueError(f"Unknown flux: {name}")


def initial_pool(seed: int, max_abs: float, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    """Reproduce five batches of 100 GP initial conditions.

    The unused random equation coefficients are intentionally retained: they
    consume RNG values in the historical generator before GP sampling.
    """
    torch.manual_seed(seed)
    params = data_utils.generate_random_parameters(
        EQNS, "random_-1_1", seed=seed, device=device
    )
    xs = np.linspace(0.0, 1.0, LENGTH, endpoint=False)
    xs_tensor = torch.tensor(xs)
    batches = []
    for equation_index in range(len(params)):
        attempts = 0
        while True:
            attempts += 1
            init = data_utils.generate_gaussian_process(
                xs_tensor,
                NUM,
                data_utils.rbf_circle_kernel_1d,
                1.0,
                1.0,
                device=device,
            )
            if torch.max(torch.abs(init)) < max_abs:
                batches.append(init.unsqueeze(-1))
                break
        print(
            f"  initial batch {equation_index + 1}/{EQNS}, attempts={attempts}",
            flush=True,
        )
    return xs, torch.cat(batches)


def capture_windows(
    init: torch.Tensor,
    starts: Sequence[int],
    path_indices: Sequence[int],
    group_len: int,
    flux_fn: Callable[[torch.Tensor], torch.Tensor],
    grad_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Evolve one trajectory bank and retain arbitrary strided windows."""
    if len(starts) != len(path_indices):
        raise ValueError("starts and path_indices must have equal length")
    if any(path < 0 or path >= len(init) for path in path_indices):
        raise ValueError("path index outside initial-condition bank")

    sample_count = len(starts)
    result = torch.empty(
        sample_count,
        group_len,
        init.shape[1],
        init.shape[2],
        dtype=init.dtype,
        device=init.device,
    )
    requests: dict[int, list[tuple[int, int, int]]] = {}
    for sample_index, (start, path) in enumerate(zip(starts, path_indices)):
        for frame_index in range(group_len):
            solver_step = int(start) + frame_index * STRIDE
            requests.setdefault(solver_step, []).append(
                (sample_index, frame_index, int(path))
            )

    last_step = max(requests)
    u_range = torch.tensor([-3, 3], dtype=torch.float32, device=init.device)
    alpha = repeat(
        weno_scheme.get_scalar_alpha(u_range, grad_fn, 100, 0.1),
        " -> b",
        b=init.shape[0],
    )
    left_bound = torch.zeros_like(init, dtype=torch.float32)
    right_bound = torch.zeros_like(init, dtype=torch.float32)
    current = init
    started_at = time.monotonic()

    with torch.inference_mode():
        for solver_step in range(last_step + 1):
            for sample_index, frame_index, path_index in requests.get(solver_step, ()):
                result[sample_index, frame_index] = current[path_index]

            if solver_step == last_step:
                break
            current = weno_scheme.weno_step(
                DT,
                DX,
                current,
                weno_scheme.get_w_classic,
                weno_roll.get_u_roll_periodic,
                flux_fn,
                alpha,
                "rk4",
                left_bound,
                right_bound,
            )
            if (solver_step + 1) % 200 == 0:
                if not torch.isfinite(current).all() or torch.max(torch.abs(current)) > 10.0:
                    raise ValueError(f"Unstable solution at solver step {solver_step + 1}")
            if (solver_step + 1) % 2000 == 0:
                elapsed = (time.monotonic() - started_at) / 60.0
                print(f"  solver step {solver_step + 1}/{last_step} ({elapsed:.1f} min)", flush=True)
    return result.cpu()


def config_snapshot(
    output_dir: Path,
    *,
    seed: int,
    max_abs: float,
    steps: int,
    group_len: int | None,
):
    data = {
        "name": "test",
        "dir": str(output_dir),
        "num": NUM,
        "eqns": EQNS,
        "length": LENGTH,
        "steps": steps,
        "dt": DT,
        "dx": DX,
        "eqn_type": "consv_cubic",
        "eqn_mode": "random_-1_1",
        "stride": [STRIDE],
        "file_split": 1,
        "truncate": None,
        "seed": seed,
        "problem_types": ["forward"],
        "init": {"kernel": "rbf_circle", "sigma": 1.0, "l": 1.0, "max_abs": max_abs},
    }
    if group_len is not None:
        # Match the field position used by the historical Hydra configuration.
        items = list(data.items())
        items.insert(11, ("group_len", group_len))
        data = dict(items)
    return OmegaConf.create(data)


def atomic_torch_save(payload: dict, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {path} or {partial}")
    torch.save(payload, partial)
    os.rename(partial, path)
    print(f"Saved {path} data={tuple(payload['data'].shape)}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_paths(output_dir: Path) -> list[Path]:
    paths = []
    for flux_name in FLUX_NAMES:
        paths.extend(
            (
                output_dir / f"test_seq_{flux_name}.pt",
                output_dir / f"test_seq_{flux_name}_150step.pt",
                output_dir / f"test_seq_{flux_name}_150step_legacy20aligned.pt",
            )
        )
    return paths


def preflight(paths: Sequence[Path]) -> None:
    collisions = [path for path in paths if path.exists() or path.with_suffix(path.suffix + ".partial").exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite:\n" + "\n".join(map(str, collisions)))


def generate_eval(output_dir: Path, device: torch.device) -> None:
    preflight(eval_paths(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_perm = permutation(EVAL_SEED, TRAJECTORIES)
    modern_rng = random.Random(EVAL_SEED)
    modern_starts = [modern_rng.randint(0, 200) for _ in range(TRAJECTORIES)]

    for flux_name in FLUX_NAMES:
        print(f"\n[{flux_name}] generating eval trajectory bank", flush=True)
        xs, init = initial_pool(EVAL_SEED, 3.0, device)
        flux_fn, grad_fn, formula = flux_functions(flux_name)

        # Both historical-window and seeded-window datasets use the same 500
        # paths, so collect both sets during one 30k-step evolution.
        starts = list(EVAL_STARTS[flux_name]) + modern_starts
        path_indices = list(range(TRAJECTORIES)) + list(range(TRAJECTORIES))
        captured = capture_windows(
            init, starts, path_indices, 150, flux_fn, grad_fn
        )
        legacy_aligned_raw = captured[:TRAJECTORIES]
        modern_raw = captured[TRAJECTORIES:]
        legacy_aligned = legacy_aligned_raw[eval_perm]
        legacy = legacy_aligned[:, :20].clone()
        modern = modern_raw[eval_perm]

        legacy_path = output_dir / f"test_seq_{flux_name}.pt"
        atomic_torch_save(
            {
                "data": legacy,
                "grid_x": torch.tensor(xs, dtype=torch.float32),
                "config": config_snapshot(
                    output_dir,
                    seed=EVAL_SEED,
                    max_abs=3.0,
                    steps=4000,
                    group_len=20,
                ),
            },
            legacy_path,
        )

        atomic_torch_save(
            {
                "data": modern,
                "grid_x": torch.tensor(xs, dtype=torch.float32),
                "config": config_snapshot(
                    output_dir,
                    seed=EVAL_SEED,
                    max_abs=3.0,
                    steps=30000,
                    group_len=150,
                ),
                "generation": {
                    "flux_type": flux_name,
                    "flux_formula": formula,
                    "torch_seed": EVAL_SEED,
                    "python_seed": EVAL_SEED,
                    "start_min": 0,
                    "start_max": 200,
                    "sampling": "one strided sequence per GP trajectory",
                    "source_script": Path(__file__).name,
                },
            },
            output_dir / f"test_seq_{flux_name}_150step.pt",
        )

        atomic_torch_save(
            {
                "data": legacy_aligned,
                "grid_x": torch.tensor(xs, dtype=torch.float32),
                "config": config_snapshot(
                    output_dir,
                    seed=EVAL_SEED,
                    max_abs=3.0,
                    steps=30000,
                    group_len=150,
                ),
                "generation": {
                    "flux_type": flux_name,
                    "flux_formula": formula,
                    "torch_seed": EVAL_SEED,
                    "legacy_source": legacy_path.name,
                    "legacy_source_sha256": sha256_file(legacy_path),
                    "legacy_prefix_frames": 20,
                    "continuation_frames": 130,
                    "continuation_solver_steps": 26000,
                    "alignment": "frames 0:20 reproduced from recovered historical starts",
                    "source_script": Path(__file__).name,
                },
            },
            output_dir / f"test_seq_{flux_name}_150step_legacy20aligned.pt",
        )
        del init, captured, legacy_aligned_raw, modern_raw, legacy_aligned, legacy, modern
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_evolve(output_dir: Path, device: torch.device) -> None:
    output_path = output_dir / "test_seq.pt"
    preflight([output_path])
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[sin] generating evolve double-window trajectory bank", flush=True)
    xs, init = initial_pool(EVOLVE_SEED, 5.0, device)
    flux_fn, grad_fn, _ = flux_functions("sin")
    path_indices = [raw_slot // 2 for raw_slot in range(2 * TRAJECTORIES)]
    raw = capture_windows(
        init,
        EVOLVE_STARTS,
        path_indices,
        7,
        flux_fn,
        grad_fn,
    )
    data = raw[permutation(EVOLVE_SEED, 2 * TRAJECTORIES)]
    atomic_torch_save(
        {
            "data": data,
            "grid_x": torch.tensor(xs, dtype=torch.float32),
            "config": config_snapshot(
                output_dir,
                seed=EVOLVE_SEED,
                max_abs=5.0,
                steps=1500,
                group_len=None,
            ),
        },
        output_path,
    )


def smoke_check(device: torch.device) -> None:
    validate_recovered_starts()
    flux_fn, grad_fn, _ = flux_functions("sin")
    init = torch.zeros((2, LENGTH, 1), dtype=torch.float64, device=device)
    sample = capture_windows(init, [0, 1], [0, 1], 1, flux_fn, grad_fn)
    if tuple(sample.shape) != (2, 1, LENGTH, 1) or not torch.isfinite(sample).all():
        raise RuntimeError("Conservation smoke check failed")
    print("Conservation generator self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("eval", "evolve"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    validate_recovered_starts()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact reproduction but is unavailable")
    device = torch.device(args.device)
    if args.check_only:
        smoke_check(device)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-only is used")

    output_dir = args.output_dir.expanduser().resolve()
    if args.dataset == "eval":
        generate_eval(output_dir, device)
    else:
        generate_evolve(output_dir, device)


if __name__ == "__main__":
    main()
