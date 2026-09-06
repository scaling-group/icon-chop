from __future__ import annotations

import numpy as np
import torch


def build_diag_block(cond_len: int, qoi_kv_len: int, qoi_k_len: int) -> torch.Tensor:
    diag_block = np.zeros((cond_len + qoi_kv_len + qoi_k_len, cond_len + qoi_kv_len + qoi_k_len), dtype=bool)
    diag_block[:, :cond_len] = 1
    diag_block[cond_len : cond_len + qoi_kv_len, cond_len : cond_len + qoi_kv_len] = 1
    diag_block[cond_len + qoi_kv_len :, cond_len + qoi_kv_len :] = np.eye(qoi_k_len, dtype=bool)
    return torch.tensor(diag_block, dtype=torch.bool)


def build_bool_sequence(demo_num: int, mode: str, shot_num_min: int):
    if mode == "train":
        cond_list = [True] * demo_num + [True]
        qoi_kv_list = [True] * demo_num + [False]
        qoi_k_list = [index >= shot_num_min for index in range(demo_num)] + [True]
    elif mode == "test":
        cond_list = [True] * demo_num + [True]
        qoi_kv_list = [True] * demo_num + [False]
        qoi_k_list = [False] * demo_num + [True]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return cond_list, qoi_kv_list, qoi_k_list


def build_basic_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list) -> torch.Tensor:
    count = len(cond_len_list)
    mask_size = sum(cond_len_list[index] + qoi_kv_len_list[index] + qoi_k_len_list[index] for index in range(count))
    mask = np.zeros((mask_size, mask_size), dtype=bool)

    for row in range(count):
        for col in range(row + 1):
            row_cursor = sum(
                cond_len_list[index] + qoi_kv_len_list[index] + qoi_k_len_list[index] for index in range(row)
            )
            row_block_size = cond_len_list[row] + qoi_kv_len_list[row] + qoi_k_len_list[row]
            col_cursor = sum(
                cond_len_list[index] + qoi_kv_len_list[index] + qoi_k_len_list[index] for index in range(col)
            )
            col_cond_len = cond_len_list[col]
            col_qoi_kv_len = qoi_kv_len_list[col]

            if row == col:
                mask[row_cursor : row_cursor + row_block_size, col_cursor : col_cursor + row_block_size] = build_diag_block(
                    cond_len_list[row],
                    qoi_kv_len_list[row],
                    qoi_k_len_list[row],
                )
            else:
                mask[row_cursor : row_cursor + row_block_size, col_cursor : col_cursor + col_cond_len + col_qoi_kv_len] = True
    return torch.tensor(mask, dtype=torch.bool)


def build_index_integer(cond_len_list, qoi_kv_len_list, qoi_k_len_list) -> torch.Tensor:
    index = []
    for block_index in range(len(cond_len_list)):
        index += [block_index * 3] * cond_len_list[block_index]
        index += [block_index * 3 + 1] * qoi_kv_len_list[block_index]
        index += [block_index * 3 + 2] * qoi_k_len_list[block_index]
    return torch.tensor(index)


def build_out_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list, num_range) -> torch.Tensor:
    count = len(cond_len_list)
    out_mask_size = sum(cond_len_list[index] + qoi_kv_len_list[index] + qoi_k_len_list[index] for index in range(count))
    out_mask = np.zeros((out_mask_size), dtype=bool)
    begin, end = num_range
    cursor = 0
    for index in range(count):
        pair_size = cond_len_list[index] + qoi_kv_len_list[index] + qoi_k_len_list[index]
        if begin <= index < end:
            out_mask[cursor + cond_len_list[index] + qoi_kv_len_list[index] : cursor + pair_size] = 1
        cursor += pair_size
    return torch.tensor(out_mask, dtype=torch.bool)


def build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list) -> torch.Tensor:
    demo_cond = torch.cat([data["demo_cond_k"], data["demo_cond_v"]], dim=-1)
    demo_qoi_kv = torch.cat([data["demo_qoi_k"], data["demo_qoi_v"]], dim=-1)
    demo_qoi_k = torch.nn.functional.pad(data["demo_qoi_k"], (0, data["demo_qoi_v"].shape[-1]))

    demo_num = data["demo_cond_k"].shape[1]
    quest_cond = torch.cat([data["quest_cond_k"], data["quest_cond_v"]], dim=-1)
    value_dim = data["demo_qoi_v"].shape[-1]
    batch_size = data["quest_cond_k"].shape[0]
    quest_qoi_len = data["quest_qoi_k"].shape[-2]
    quest_qoi_v = torch.zeros((batch_size, 1, quest_qoi_len, value_dim), device=data["demo_qoi_v"].device)
    quest_qoi_kv = torch.cat([data["quest_qoi_k"], quest_qoi_v], dim=-1)
    quest_qoi_k = torch.nn.functional.pad(data["quest_qoi_k"], (0, value_dim))

    sequence = []
    for index in range(demo_num):
        if cond_bool_list[index]:
            sequence.append(demo_cond[:, index])
        if qoi_kv_bool_list[index]:
            sequence.append(demo_qoi_kv[:, index])
        if qoi_k_bool_list[index]:
            sequence.append(demo_qoi_k[:, index])

    if cond_bool_list[-1]:
        sequence.append(quest_cond[:, 0])
    if qoi_kv_bool_list[-1]:
        sequence.append(quest_qoi_kv[:, 0])
    if qoi_k_bool_list[-1]:
        sequence.append(quest_qoi_k[:, 0])

    return torch.cat(sequence, dim=1)


def build_matrices(data_shape, mode: str, shot_num_min: int):
    demo_num = data_shape["demo_cond_k"][0]
    demo_cond_len = data_shape["demo_cond_k"][1]
    demo_qoi_len = data_shape["demo_qoi_k"][1]
    quest_cond_len = data_shape["quest_cond_k"][-2]
    quest_qoi_len = data_shape["quest_qoi_k"][-2]

    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = build_bool_sequence(demo_num, mode, shot_num_min)
    cond_len_list = [demo_cond_len if flag else 0 for flag in cond_bool_list[:-1]] + [quest_cond_len if cond_bool_list[-1] else 0]
    qoi_kv_len_list = [demo_qoi_len if flag else 0 for flag in qoi_kv_bool_list[:-1]] + [quest_qoi_len if qoi_kv_bool_list[-1] else 0]
    qoi_k_len_list = [demo_qoi_len if flag else 0 for flag in qoi_k_bool_list[:-1]] + [quest_qoi_len if qoi_k_bool_list[-1] else 0]

    basic_mask = build_basic_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list)
    index_pos = build_index_integer(cond_len_list, qoi_kv_len_list, qoi_k_len_list)
    out_mask = build_out_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list, (shot_num_min, demo_num + 1))
    return basic_mask, index_pos, out_mask
