from __future__ import annotations

import copy

import einops
import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Linear, ModuleList

from . import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout2 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout3 = Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights: bool = False):
        attn_out, weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
        )
        src = self.norm1(src + self.dropout1(attn_out))

        ff_out = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout3(ff_out))

        if need_weights:
            return src, weights
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None, need_weights: bool = False):
        weights = []
        for layer in self.layers:
            if need_weights:
                src, attn = layer(src, mask, src_key_padding_mask, need_weights=True)
                weights.append(attn)
            else:
                src = layer(src, mask, src_key_padding_mask, need_weights=False)
        if need_weights:
            return src, weights
        return src


class ICON(nn.Module):
    def __init__(
        self,
        shot_num_min: int = 1,
        data_mask: bool = False,
        in_features: int = 3,
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
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.post_projection = nn.Linear(d_model, out_features)

        self.basic_mask = {}
        self.index_pos = {}
        self.out_mask = {}

    def _get_matrices(self, data_shape, mode: str, shot_num_min: int):
        key = (tuple(data_shape.items()), mode, shot_num_min)
        if key not in self.basic_mask:
            basic_mask, index_pos, out_mask = utils.build_matrices(data_shape, mode=mode, shot_num_min=shot_num_min)
            device = next(self.parameters()).device
            self.basic_mask[key] = basic_mask.to(device)
            self.index_pos[key] = index_pos.to(device)
            self.out_mask[key] = out_mask.to(device)
        return self.basic_mask[key], self.index_pos[key], self.out_mask[key]

    def forward(self, data, mode: str = "test", need_weights: bool = False, **kwargs):
        data_shape = {key: value.shape[1:] for key, value in data.items() if isinstance(value, torch.Tensor)}
        shot_num = self.shot_num_min if mode == "train" else 0
        basic_mask, index_pos, out_mask = self._get_matrices(data_shape, mode, shot_num)

        demo_num = data["demo_cond_k"].shape[1]
        cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = utils.build_bool_sequence(demo_num, mode=mode, shot_num_min=shot_num)
        sequence = utils.build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list)

        hidden = self.pre_projection(sequence)
        hidden = hidden + self.function_pe(index_pos)

        if need_weights:
            hidden, weights = self.transformer(hidden, mask=~basic_mask, src_key_padding_mask=None, need_weights=True)
        else:
            hidden = self.transformer(hidden, mask=~basic_mask, src_key_padding_mask=None, need_weights=False)
        hidden = self.post_projection(hidden)

        if mode == "train":
            hidden = hidden[:, out_mask, :]
            hidden = einops.rearrange(hidden, "b (n qoi_len) d -> b n qoi_len d", qoi_len=data["demo_qoi_v"].shape[-2])
        else:
            quest_qoi_len = data["quest_qoi_k"].shape[-2]
            hidden = hidden[:, None, -quest_qoi_len:, :]

        if need_weights:
            return hidden, weights
        return hidden
