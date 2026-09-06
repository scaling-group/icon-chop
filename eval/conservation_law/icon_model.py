"""Self-contained ICON model and inference wrapper for conservation-law rollouts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn


def _build_diag_block(cond_len: int, qoi_kv_len: int, qoi_k_len: int) -> torch.Tensor:
    size = cond_len + qoi_kv_len + qoi_k_len
    block = np.zeros((size, size), dtype=bool)
    block[:, :cond_len] = True
    block[cond_len : cond_len + qoi_kv_len, cond_len : cond_len + qoi_kv_len] = True
    block[cond_len + qoi_kv_len :, cond_len + qoi_kv_len :] = np.eye(qoi_k_len, dtype=bool)
    return torch.as_tensor(block, dtype=torch.bool)


def _build_bool_sequence(demo_num: int, mode: str, shot_num_min: int):
    if mode == "train":
        cond = [True] * demo_num + [True]
        qoi_kv = [True] * demo_num + [False]
        qoi_k = [index >= shot_num_min for index in range(demo_num)] + [True]
    elif mode == "test":
        cond = [True] * demo_num + [True]
        qoi_kv = [True] * demo_num + [False]
        qoi_k = [False] * demo_num + [True]
    else:
        raise ValueError(f"Unsupported ICON mode: {mode}")
    return cond, qoi_kv, qoi_k


def _build_basic_mask(cond_lens, qoi_kv_lens, qoi_k_lens) -> torch.Tensor:
    count = len(cond_lens)
    mask_size = sum(cond_lens[i] + qoi_kv_lens[i] + qoi_k_lens[i] for i in range(count))
    mask = torch.zeros((mask_size, mask_size), dtype=torch.bool)
    row_cursor = 0
    for row in range(count):
        row_size = cond_lens[row] + qoi_kv_lens[row] + qoi_k_lens[row]
        col_cursor = 0
        for col in range(row + 1):
            col_cond_qoi = cond_lens[col] + qoi_kv_lens[col]
            col_size = col_cond_qoi + qoi_k_lens[col]
            if row == col:
                mask[row_cursor : row_cursor + row_size, col_cursor : col_cursor + col_size] = _build_diag_block(
                    cond_lens[row], qoi_kv_lens[row], qoi_k_lens[row]
                )
            else:
                mask[row_cursor : row_cursor + row_size, col_cursor : col_cursor + col_cond_qoi] = True
            col_cursor += col_size
        row_cursor += row_size
    return mask


def _build_index_integer(cond_lens, qoi_kv_lens, qoi_k_lens) -> torch.Tensor:
    indices: list[int] = []
    for block_index in range(len(cond_lens)):
        indices.extend([block_index * 3] * cond_lens[block_index])
        indices.extend([block_index * 3 + 1] * qoi_kv_lens[block_index])
        indices.extend([block_index * 3 + 2] * qoi_k_lens[block_index])
    return torch.as_tensor(indices, dtype=torch.long)


def _build_out_mask(cond_lens, qoi_kv_lens, qoi_k_lens, begin: int, end: int) -> torch.Tensor:
    total = sum(cond_lens[i] + qoi_kv_lens[i] + qoi_k_lens[i] for i in range(len(cond_lens)))
    out_mask = torch.zeros(total, dtype=torch.bool)
    cursor = 0
    for index in range(len(cond_lens)):
        pair_size = cond_lens[index] + qoi_kv_lens[index] + qoi_k_lens[index]
        if begin <= index < end:
            start = cursor + cond_lens[index] + qoi_kv_lens[index]
            out_mask[start : cursor + pair_size] = True
        cursor += pair_size
    return out_mask


def _build_matrices(data_shape, mode: str, shot_num_min: int):
    demo_num = data_shape["demo_cond_k"][0]
    demo_cond_len = data_shape["demo_cond_k"][1]
    demo_qoi_len = data_shape["demo_qoi_k"][1]
    quest_cond_len = data_shape["quest_cond_k"][-2]
    quest_qoi_len = data_shape["quest_qoi_k"][-2]
    cond_flags, qoi_kv_flags, qoi_k_flags = _build_bool_sequence(demo_num, mode, shot_num_min)
    cond_lens = [demo_cond_len if flag else 0 for flag in cond_flags[:-1]] + [
        quest_cond_len if cond_flags[-1] else 0
    ]
    qoi_kv_lens = [demo_qoi_len if flag else 0 for flag in qoi_kv_flags[:-1]] + [
        quest_qoi_len if qoi_kv_flags[-1] else 0
    ]
    qoi_k_lens = [demo_qoi_len if flag else 0 for flag in qoi_k_flags[:-1]] + [
        quest_qoi_len if qoi_k_flags[-1] else 0
    ]
    return (
        _build_basic_mask(cond_lens, qoi_kv_lens, qoi_k_lens),
        _build_index_integer(cond_lens, qoi_kv_lens, qoi_k_lens),
        _build_out_mask(cond_lens, qoi_kv_lens, qoi_k_lens, shot_num_min, demo_num + 1),
    )


def _build_data_sequence(data, cond_flags, qoi_kv_flags, qoi_k_flags) -> torch.Tensor:
    demo_cond = torch.cat([data["demo_cond_k"], data["demo_cond_v"]], dim=-1)
    demo_qoi_kv = torch.cat([data["demo_qoi_k"], data["demo_qoi_v"]], dim=-1)
    demo_qoi_k = torch.nn.functional.pad(data["demo_qoi_k"], (0, data["demo_qoi_v"].shape[-1]))
    quest_cond = torch.cat([data["quest_cond_k"], data["quest_cond_v"]], dim=-1)
    quest_qoi_k = torch.nn.functional.pad(data["quest_qoi_k"], (0, data["demo_qoi_v"].shape[-1]))

    sequence = []
    for index in range(data["demo_cond_k"].shape[1]):
        if cond_flags[index]:
            sequence.append(demo_cond[:, index])
        if qoi_kv_flags[index]:
            sequence.append(demo_qoi_kv[:, index])
        if qoi_k_flags[index]:
            sequence.append(demo_qoi_k[:, index])
    if cond_flags[-1]:
        sequence.append(quest_cond[:, 0])
    if qoi_kv_flags[-1]:
        zeros = torch.zeros_like(data["quest_qoi_k"])
        sequence.append(torch.cat([data["quest_qoi_k"][:, 0], zeros[:, 0]], dim=-1))
    if qoi_k_flags[-1]:
        sequence.append(quest_qoi_k[:, 0])
    return torch.cat(sequence, dim=1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attended, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(src + self.dropout1(attended))
        feedforward = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout3(feedforward))


class TransformerEncoder(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src


class ICON(nn.Module):
    """The exact 1D ICON architecture stored in ``conservation.ckpt``."""

    def __init__(
        self,
        shot_num_min: int = 1,
        data_mask: bool = False,
        in_features: int = 2,
        out_features: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        num_embeddings: int = 100,
    ):
        super().__init__()
        self.shot_num_min = shot_num_min
        self.data_mask = data_mask
        self.pre_projection = nn.Linear(in_features, d_model)
        self.function_pe = nn.Embedding(num_embeddings, d_model)
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(layer, num_layers)
        self.post_projection = nn.Linear(d_model, out_features)
        self._matrix_cache = {}

    def _get_matrices(self, data_shape, mode: str, shot_num_min: int):
        key = (tuple(data_shape.items()), mode, shot_num_min)
        if key not in self._matrix_cache:
            device = next(self.parameters()).device
            self._matrix_cache[key] = tuple(item.to(device) for item in _build_matrices(data_shape, mode, shot_num_min))
        return self._matrix_cache[key]

    def forward(self, data, mode: str = "test"):
        data_shape = {key: value.shape[1:] for key, value in data.items() if isinstance(value, torch.Tensor)}
        shot_num = self.shot_num_min if mode == "train" else 0
        basic_mask, index_pos, out_mask = self._get_matrices(data_shape, mode, shot_num)
        demo_num = data["demo_cond_k"].shape[1]
        flags = _build_bool_sequence(demo_num, mode, shot_num)
        hidden = self.pre_projection(_build_data_sequence(data, *flags))
        hidden = hidden + self.function_pe(index_pos)
        hidden = self.transformer(hidden, mask=~basic_mask, src_key_padding_mask=None)
        hidden = self.post_projection(hidden)
        if mode == "train":
            qoi_len = data["demo_qoi_v"].shape[-2]
            selected = hidden[:, out_mask, :]
            return selected.reshape(selected.shape[0], -1, qoi_len, selected.shape[-1])
        quest_qoi_len = data["quest_qoi_k"].shape[-2]
        return hidden[:, None, -quest_qoi_len:, :]


def load_icon_model(checkpoint_path: str | Path, device: torch.device) -> ICON:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    cleaned = {key.removeprefix("net."): value for key, value in state_dict.items()}
    model = ICON()
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


class IconWrapper:
    """Translate the evaluator's semantic keys to ICON's model keys."""

    def __init__(self, model: ICON):
        self.model = model

    @torch.inference_mode()
    def predict(self, transform=None, **kwargs) -> torch.Tensor:
        data: Mapping[str, torch.Tensor] = kwargs["data"] if "data" in kwargs else kwargs
        device = next(self.model.parameters()).device
        demo_input_vals = data["demo_input_vals"].to(device)
        demo_target_vals = data["demo_target_vals"].to(device)
        query_input_vals = data["query_input_vals"].to(device)
        if transform is not None:
            demo_input_vals = transform.forward(demo_input_vals)
            demo_target_vals = transform.forward(demo_target_vals)
            query_input_vals = transform.forward(query_input_vals)

        demo_input_coords = data["demo_input_coords"].to(device)
        demo_target_coords = data["demo_target_coords"].to(device)
        query_input_coords = data["query_input_coords"].to(device)
        query_target_coords = data["query_target_coords"].to(device)
        model_data = {
            "demo_cond_k": demo_input_coords,
            "demo_cond_v": demo_input_vals,
            "demo_cond_mask": torch.ones_like(demo_input_vals[..., 0], dtype=torch.bool),
            "demo_qoi_k": demo_target_coords,
            "demo_qoi_v": demo_target_vals,
            "demo_qoi_mask": torch.ones_like(demo_target_vals[..., 0], dtype=torch.bool),
            "quest_cond_k": query_input_coords,
            "quest_cond_v": query_input_vals,
            "quest_cond_mask": torch.ones_like(query_input_vals[..., 0], dtype=torch.bool),
            "quest_qoi_k": query_target_coords,
            "quest_qoi_mask": torch.ones_like(query_target_coords[..., 0], dtype=torch.bool),
            "quest_qoi_v": torch.zeros_like(query_target_coords),
        }
        output = self.model(model_data, mode="test")[:, 0]
        return transform.backward(output) if transform is not None else output

