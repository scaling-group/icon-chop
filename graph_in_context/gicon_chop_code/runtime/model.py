from __future__ import annotations

from pathlib import Path
from typing import Mapping

import hydra
import torch
from omegaconf import OmegaConf


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device {device_name!r}, but CUDA is unavailable")
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_gicon_model(ckpt_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = checkpoint["hyper_parameters"]["cfg"]
    model_cfg = _localize_model_config(cfg.model)
    model = hydra.utils.instantiate(model_cfg)
    state_dict = {
        key.removeprefix("net."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("net.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _localize_model_config(model_cfg):
    cfg = OmegaConf.to_container(model_cfg, resolve=True)

    def visit(value):
        if isinstance(value, dict):
            target = value.get("_target_")
            if isinstance(target, str):
                value["_target_"] = _local_target(target)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(cfg)
    return OmegaConf.create(cfg)


def _local_target(target: str) -> str:
    source_prefix = "src" + ".models.gicon."
    replacements = {
        source_prefix + "gicon_frames.": "runtime.modeling.gicon_frames.",
        source_prefix + "gicon_utils.": "runtime.modeling.gicon_utils.",
        source_prefix + "gicon_attn.": "runtime.modeling.gicon_attn.",
    }
    for old, new in replacements.items():
        if target.startswith(old):
            return target.replace(old, new, 1)
    return target


class GiconWrapper:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def predict(self, *, data: Mapping[str, torch.Tensor], graph) -> torch.Tensor:
        device = self.device
        model_data = {key: value.to(device) for key, value in data.items()}
        output = self.model(model_data, graph=graph)
        return output["quest_pred_v"]
