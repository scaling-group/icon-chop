"""Recover historical random window starts from the preserved .pt files.

This is a verification/forensics utility.  Normal regeneration uses the
checked-in constants in ``window_starts.py`` and does not need the old files.
The historical generators seeded Torch but did not seed Python's ``random``;
we therefore identify every saved first frame against the seed-reconstructed
trajectory bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from einops import repeat

import data_utils
from weno_new import weno_roll, weno_scheme


def flux_functions(name: str) -> tuple[Callable, Callable]:
    if name == "sin":
        return (
            lambda u: torch.sin(u) - torch.cos(u),
            lambda u: torch.cos(u) + torch.sin(u),
        )
    if name == "tanh":
        return lambda u: torch.tanh(u), lambda u: 1.0 - torch.tanh(u) ** 2
    if name == "u2":
        return (
            lambda u: u**2 / (u**2 + (1.0 - u) ** 2),
            lambda u: 2.0 * u * (1.0 - u) / (u**2 + (1.0 - u) ** 2) ** 2,
        )
    raise ValueError(name)


def initial_pool(seed: int, max_abs: float) -> torch.Tensor:
    torch.manual_seed(seed)
    params = data_utils.generate_random_parameters(5, "random_-1_1", seed=seed)
    xs = torch.tensor(np.linspace(0.0, 1.0, 100, endpoint=False))
    batches = []
    for _ in params:
        while True:
            init = data_utils.generate_gaussian_process(
                xs, 100, data_utils.rbf_circle_kernel_1d, 1.0, 1.0
            )
            if torch.max(torch.abs(init)) < max_abs:
                batches.append(init.unsqueeze(-1))
                break
    return torch.cat(batches)


def byte_key(row: torch.Tensor) -> bytes:
    return row.detach().contiguous().cpu().numpy().tobytes()


def recover(
    path: Path,
    *,
    seed: int,
    max_abs: float,
    flux_name: str,
    max_start: int,
    windows_per_path: int,
) -> list[int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    target = payload["data"][:, 0, :, 0].contiguous()
    expected_count = 500 * windows_per_path
    if tuple(target.shape) != (expected_count, 100):
        raise ValueError(f"Unexpected target shape in {path}: {tuple(target.shape)}")

    target_lookup: dict[bytes, list[int]] = {}
    for saved_index, row in enumerate(target):
        target_lookup.setdefault(byte_key(row), []).append(saved_index)

    current = initial_pool(seed, max_abs)
    flux_fn, grad_fn = flux_functions(flux_name)
    u_range = torch.tensor([-3, 3], dtype=torch.float32, device=current.device)
    alpha = repeat(
        weno_scheme.get_scalar_alpha(u_range, grad_fn, 100, 0.1),
        " -> b",
        b=current.shape[0],
    )
    left_bound = torch.zeros_like(current, dtype=torch.float32)
    right_bound = torch.zeros_like(current, dtype=torch.float32)

    found: dict[int, tuple[int, int]] = {}
    with torch.inference_mode():
        for start_t in range(max_start + 1):
            current_cpu = current.detach().cpu()
            for trajectory_index, row in enumerate(current_cpu[:, :, 0]):
                for saved_index in target_lookup.get(byte_key(row), []):
                    previous = found.get(saved_index)
                    candidate = (trajectory_index, start_t)
                    if previous is not None and previous != candidate:
                        raise RuntimeError(
                            f"Non-unique match for saved sample {saved_index}: "
                            f"{previous} and {candidate}"
                        )
                    found[saved_index] = candidate

            if start_t != max_start:
                current = weno_scheme.weno_step(
                    0.0005,
                    0.01,
                    current,
                    weno_scheme.get_w_classic,
                    weno_roll.get_u_roll_periodic,
                    flux_fn,
                    alpha,
                    "rk4",
                    left_bound,
                    right_bound,
                )

    if len(found) != expected_count:
        missing = sorted(set(range(expected_count)) - set(found))
        raise RuntimeError(f"Matched {len(found)}/{expected_count}; missing {missing[:10]}")

    # The historical randperm is the first CPU RNG draw after manual_seed;
    # all initial-condition draws occurred on CUDA.
    torch.manual_seed(seed)
    permutation = torch.randperm(expected_count).tolist()
    raw_starts: list[int | None] = [None] * expected_count
    for saved_index, raw_slot in enumerate(permutation):
        trajectory_index, start_t = found[saved_index]
        if trajectory_index != raw_slot // windows_per_path:
            raise RuntimeError(
                f"Permutation mismatch at saved={saved_index}: raw={raw_slot}, "
                f"trajectory={trajectory_index}"
            )
        raw_starts[raw_slot] = start_t

    if any(value is None for value in raw_starts):
        raise RuntimeError("Internal error: incomplete raw start list")
    return [int(value) for value in raw_starts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory containing the conservation-law evaluation datasets.",
    )
    parser.add_argument(
        "--evolve-dir",
        type=Path,
        required=True,
        help="Directory containing the conservation-law evolution dataset.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Exact recovery requires the CUDA environment used originally")

    result = {
        "eval": {
            flux: recover(
                args.eval_dir / f"test_seq_{flux}.pt",
                seed=1234567890123,
                max_abs=3.0,
                flux_name=flux,
                max_start=200,
                windows_per_path=1,
            )
            for flux in ("sin", "tanh", "u2")
        },
        "evolve": recover(
            args.evolve_dir / "test_seq.pt",
            seed=123456789,
            max_abs=5.0,
            flux_name="sin",
            max_start=300,
            windows_per_path=2,
        ),
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
