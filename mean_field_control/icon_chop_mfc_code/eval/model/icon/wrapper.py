from __future__ import annotations

import torch


class IconWrapper:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def predict(self, transform=None, **kwargs):
        data = kwargs["data"] if "data" in kwargs else kwargs
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

        demo_cond_mask = torch.ones_like(demo_input_vals[..., 0], dtype=torch.bool, device=device)
        demo_qoi_mask = torch.ones_like(demo_target_vals[..., 0], dtype=torch.bool, device=device)
        query_cond_mask = torch.ones_like(query_input_vals[..., 0], dtype=torch.bool, device=device)
        query_qoi_mask = torch.ones_like(query_target_coords[..., 0], dtype=torch.bool, device=device)

        model_data = {
            "demo_cond_k": demo_input_coords,
            "demo_cond_v": demo_input_vals,
            "demo_cond_mask": demo_cond_mask,
            "demo_qoi_k": demo_target_coords,
            "demo_qoi_v": demo_target_vals,
            "demo_qoi_mask": demo_qoi_mask,
            "quest_cond_k": query_input_coords,
            "quest_cond_v": query_input_vals,
            "quest_cond_mask": query_cond_mask,
            "quest_qoi_k": query_target_coords,
            "quest_qoi_mask": query_qoi_mask,
            "quest_qoi_v": torch.zeros_like(query_target_coords, device=device),
        }

        prediction = self.model.forward(model_data, mode="test")
        output = prediction[:, 0]
        if transform is not None:
            output = transform.backward(output)
        return output
