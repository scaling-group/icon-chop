from __future__ import annotations

import glob
import logging
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .mfc_icon import resolve_subsample_indices

log = logging.getLogger(__name__)


class MfcRawOperatorDataset(Dataset):
    def __init__(
        self,
        file_paths: str | list[str],
        max_len: int | None,
        k_dim: int,
        base_seed: int | None = None,
        time_subsample: int | None = None,
        space_subsample: int | None = None,
    ) -> None:
        if isinstance(file_paths, str):
            self.file_paths = sorted(glob.glob(file_paths))
        else:
            self.file_paths = sorted(file_paths)

        log.info("%s files found in %s: %s", len(self.file_paths), file_paths, self.file_paths)

        self.max_len = max_len
        self.k_dim = k_dim
        self.base_seed = base_seed
        self.time_subsample = time_subsample
        self.space_subsample = space_subsample
        self.indices: list[tuple[str, str]] = []
        self.file_handles: dict[str, h5py.File] = {}

        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                log.warning("File %s does not exist, skipping...", file_path)
                continue
            try:
                with h5py.File(file_path, "r") as file_handle:
                    self.indices.extend((file_path, key) for key in file_handle)
            except Exception as error:
                log.warning("Could not read file %s: %s", file_path, error)

    def __len__(self) -> int:
        return len(self.indices)

    def _make_rng(self, idx: int) -> torch.Generator | None:
        if self.base_seed is None:
            return None
        rng = torch.Generator()
        rng.manual_seed(self.base_seed + idx)
        return rng

    def _randperm(self, length: int, rng: torch.Generator | None) -> torch.Tensor:
        if rng is None:
            return torch.randperm(length)
        return torch.randperm(length, generator=rng)

    def _pad_k(self, coord: torch.Tensor) -> torch.Tensor:
        coord_dim = coord.shape[-1]
        if coord_dim > self.k_dim:
            raise ValueError(f"Coordinate dimension {coord_dim} exceeds configured k_dim={self.k_dim}.")
        if coord_dim == self.k_dim:
            return coord

        pad = torch.zeros(*coord.shape[:-1], self.k_dim - coord_dim, dtype=coord.dtype)
        return torch.cat([pad, coord], dim=-1)

    def _subsample(
        self,
        coord: torch.Tensor,
        value: torch.Tensor,
        _rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = resolve_subsample_indices(
            coord,
            max_len=self.max_len,
            time_subsample=self.time_subsample,
            space_subsample=self.space_subsample,
        )
        if indices is None:
            raise ValueError(
                f"Could not build a deterministic downsample for coordinates with shape {tuple(coord.shape)} "
                f"(max_len={self.max_len}, time_subsample={self.time_subsample}, "
                f"space_subsample={self.space_subsample})."
            )
        if indices.numel() == coord.shape[1]:
            return coord, value
        return coord[:, indices, :], value[:, indices, :]

    def __getitem__(self, idx: int) -> dict[str, object]:
        rng = self._make_rng(idx)
        file_path, group_name = self.indices[idx]

        if file_path not in self.file_handles:
            self.file_handles[file_path] = h5py.File(file_path, "r")

        group = self.file_handles[file_path][group_name]
        equation = group["equation"][()].decode("utf-8")

        cond_k = torch.tensor(group["cond_k"][:], dtype=torch.float32)
        cond_v = torch.tensor(group["cond_v"][:], dtype=torch.float32)
        qoi_k = torch.tensor(group["qoi_k"][:], dtype=torch.float32)
        qoi_v = torch.tensor(group["qoi_v"][:], dtype=torch.float32)

        num_pairs = cond_k.shape[0]
        if num_pairs < 2:
            raise ValueError(f"Raw operator dataset requires at least 2 pairs, but only {num_pairs} are available.")

        cond_k, cond_v = self._subsample(cond_k, cond_v, rng)
        qoi_k, qoi_v = self._subsample(qoi_k, qoi_v, rng)

        cond_k = self._pad_k(cond_k)
        qoi_k = self._pad_k(qoi_k)

        return {
            "description": np.array([equation], dtype=np.dtypes.StringDType()),
            "data": {
                "candidate_input_coords": cond_k.unsqueeze(0),
                "candidate_input_vals": cond_v.unsqueeze(0),
                "candidate_target_coords": qoi_k.unsqueeze(0),
                "candidate_target_vals": qoi_v.unsqueeze(0),
            },
            "pair_count": int(num_pairs),
        }

    def __del__(self) -> None:
        for file_handle in self.file_handles.values():
            if file_handle is not None:
                file_handle.close()
