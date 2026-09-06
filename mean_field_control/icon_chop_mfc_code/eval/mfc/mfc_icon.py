from __future__ import annotations

import glob
import logging
import os
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def _uniform_axis_indices(axis_size: int, target_size: int) -> torch.Tensor | None:
    if axis_size < target_size:
        return None

    indices = torch.round(torch.linspace(0, axis_size - 1, steps=target_size)).to(dtype=torch.long)
    if torch.unique_consecutive(indices).numel() != target_size:
        return None
    return indices


def _resolve_grid_sample_shape(max_len: int) -> tuple[int, int] | None:
    grid_side = int(round(max_len**0.5))
    if grid_side * grid_side != max_len:
        return None
    return grid_side, grid_side


def _active_coord_dims(coord: torch.Tensor) -> torch.Tensor:
    if coord.dim() != 3 or coord.shape[-1] == 0:
        return torch.empty(0, dtype=torch.long)

    reference = coord[0]
    active = []
    for dim in range(reference.shape[-1]):
        if torch.unique(reference[:, dim], sorted=True).numel() > 1:
            active.append(dim)
    return torch.tensor(active, dtype=torch.long)


def _uniform_line_indices(coord: torch.Tensor, target_size: int) -> torch.Tensor | None:
    if coord.dim() != 3 or coord.shape[0] == 0:
        return None

    active_dims = _active_coord_dims(coord)
    if active_dims.numel() == 0:
        return torch.arange(coord.shape[1], dtype=torch.long)
    if active_dims.numel() != 1:
        return None

    length = coord.shape[1]
    if length <= target_size:
        return torch.arange(length, dtype=torch.long)
    return _uniform_axis_indices(length, target_size)


def _structured_grid_indices(
    coord: torch.Tensor,
    max_len: int | None = None,
    time_target: int | None = None,
    space_target: int | None = None,
) -> torch.Tensor | None:
    if coord.dim() != 3 or coord.shape[0] == 0:
        return None

    active_dims = _active_coord_dims(coord)
    if active_dims.numel() < 2:
        if max_len is None:
            return None
        return _uniform_line_indices(coord, max_len)

    if active_dims.numel() > 2:
        raise ValueError(
            f"Expected at most 2 active coordinate dimensions, but found {int(active_dims.numel())} "
            f"for coordinates with shape {tuple(coord.shape)}."
        )

    if time_target is not None or space_target is not None:
        if time_target is None or space_target is None:
            raise ValueError("time_target and space_target must be provided together.")
        target_size = int(time_target) * int(space_target)
    else:
        if max_len is None:
            raise ValueError("Either max_len or (time_target, space_target) must be provided.")
        if coord.shape[1] <= max_len:
            return torch.arange(coord.shape[1], dtype=torch.long)
        grid_shape = _resolve_grid_sample_shape(max_len)
        if grid_shape is None:
            return None
        time_target, space_target = grid_shape
        target_size = max_len

    if coord.shape[1] <= target_size:
        return torch.arange(coord.shape[1], dtype=torch.long)

    reference = coord[0][:, active_dims]
    time_values = reference[:, 0]
    space_values = reference[:, 1]
    unique_t, inverse_t = torch.unique(time_values, sorted=True, return_inverse=True)
    unique_x, inverse_x = torch.unique(space_values, sorted=True, return_inverse=True)
    num_t = int(unique_t.numel())
    num_x = int(unique_x.numel())

    if num_t * num_x != reference.shape[0]:
        return None

    time_indices = _uniform_axis_indices(num_t, time_target)
    space_indices = _uniform_axis_indices(num_x, space_target)
    if time_indices is None or space_indices is None:
        return None

    grid_lookup = torch.full((num_t, num_x), -1, dtype=torch.long)
    point_indices = torch.arange(reference.shape[0], dtype=torch.long)
    grid_lookup[inverse_t, inverse_x] = point_indices
    if (grid_lookup < 0).any():
        return None

    selected = grid_lookup[time_indices][:, space_indices].reshape(-1)
    if selected.numel() != target_size:
        return None
    return selected


def resolve_subsample_indices(
    coord: torch.Tensor,
    max_len: int | None,
    time_subsample: int | None = None,
    space_subsample: int | None = None,
) -> torch.Tensor | None:
    if coord.dim() != 3 or coord.shape[0] == 0:
        return None

    active_dims = _active_coord_dims(coord)
    if active_dims.numel() < 2:
        if max_len is None:
            return torch.arange(coord.shape[1], dtype=torch.long)
        return _uniform_line_indices(coord, max_len)

    return _structured_grid_indices(
        coord,
        max_len=max_len,
        time_target=time_subsample,
        space_target=space_subsample,
    )


class MfcIconDataset(Dataset):
    def __init__(
        self,
        file_paths: str | list[str],
        demo_num: int,
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

        self.demo_num = demo_num
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
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
        if self.demo_num + 1 > num_pairs:
            raise ValueError(
                f"demo_num={self.demo_num} requires at least {self.demo_num + 1} pairs, "
                f"but only {num_pairs} are available in {group_name}."
            )

        random_indices = self._randperm(num_pairs, rng)
        demo_indices = random_indices[: self.demo_num]
        query_indices = random_indices[self.demo_num : self.demo_num + 1]

        demo_cond_k, demo_cond_v = self._subsample(cond_k[demo_indices], cond_v[demo_indices], rng)
        demo_qoi_k, demo_qoi_v = self._subsample(qoi_k[demo_indices], qoi_v[demo_indices], rng)
        query_cond_k, query_cond_v = self._subsample(cond_k[query_indices], cond_v[query_indices], rng)
        query_qoi_k, query_qoi_v = self._subsample(qoi_k[query_indices], qoi_v[query_indices], rng)

        demo_cond_k = self._pad_k(demo_cond_k)
        demo_qoi_k = self._pad_k(demo_qoi_k)
        query_cond_k = self._pad_k(query_cond_k)
        query_qoi_k = self._pad_k(query_qoi_k)

        return {
            "description": np.array([equation], dtype=np.dtypes.StringDType()),
            "data": {
                "demo_cond_k": demo_cond_k.unsqueeze(0),
                "demo_cond_v": demo_cond_v.unsqueeze(0),
                "demo_qoi_k": demo_qoi_k.unsqueeze(0),
                "demo_qoi_v": demo_qoi_v.unsqueeze(0),
                "quest_cond_k": query_cond_k.unsqueeze(0),
                "quest_cond_v": query_cond_v.unsqueeze(0),
                "quest_qoi_k": query_qoi_k.unsqueeze(0),
            },
            "label": query_qoi_v.unsqueeze(0),
        }

    def __del__(self) -> None:
        for file_handle in self.file_handles.values():
            if file_handle is not None:
                file_handle.close()
