from __future__ import annotations

import torch

from .model import ICON


def load_model(ckpt_path: str, device: str | torch.device = "cpu", **kwargs) -> ICON:
    config = {
        "shot_num_min": 1,
        "data_mask": False,
        "in_features": 3,
        "out_features": 1,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "num_embeddings": 100,
    }
    config.update(kwargs)

    model = ICON(**config)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key[4:] if key.startswith("net.") else key] = value

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model
