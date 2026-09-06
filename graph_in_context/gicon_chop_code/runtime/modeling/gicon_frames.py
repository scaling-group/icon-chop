import copy

import einops
import torch
import torch.nn as nn

from typing import Any


class InContextMessagePassingLayer(nn.Module):
    """GNN Message Passing Layer with per-node in-context learning (v-only)."""

    def __init__(self, node_dim: int, edge_dim: int, message_dim: int, incontext_transformer: nn.Module):
        super().__init__()
        self.edge_dim = edge_dim

        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, node_dim),  # Project back to node_dim
        )

        # In-context transformer for in-context learning
        self.incontext_transformer = incontext_transformer

    def _message_passing(self, node_features, edge_index, edge_features=None):
        """
        Perform message passing aggregation using einops for clear tensor operations.

        Args:
            node_features: [batch_size, seq_len, num_nodes, node_dim]
            edge_index: [2, num_edges] - source and target node indices
            edge_features: [num_edges, edge_dim] - edge features (optional)

        Returns:
            updated_sequence: [batch_size, seq_len, num_nodes, node_dim]
        """
        batch_size, seq_len, num_nodes, node_dim = node_features.shape
        num_edges = edge_index.shape[1]
        src_idx, tgt_idx = edge_index[0], edge_index[1]

        # Extract source and target nodes using advanced indexing
        src_nodes = node_features[:, :, src_idx, :]  # [batch_size, seq_len, num_edges, node_dim]
        tgt_nodes = node_features[:, :, tgt_idx, :]  # [batch_size, seq_len, num_edges, node_dim]

        # Expand edge features to match batch and sequence dimensions using einops
        if edge_features is not None:
            edge_features_expanded = einops.repeat(
                edge_features,
                "num_edges edge_dim -> batch_size seq_len num_edges edge_dim",
                batch_size=batch_size,
                seq_len=seq_len,
            )
            message_input = torch.cat([src_nodes, tgt_nodes, edge_features_expanded], dim=-1)
        else:
            edge_features_zero = torch.zeros(
                batch_size, seq_len, num_edges, self.edge_dim, device=node_features.device, dtype=node_features.dtype
            )
            message_input = torch.cat([src_nodes, tgt_nodes, edge_features_zero], dim=-1)

        # Compute messages in parallel across all batch and sequence dimensions
        messages = self.message_mlp(message_input)  # [batch_size, seq_len, num_edges, node_dim]

        # Vectorized message aggregation
        aggregated = torch.zeros_like(node_features)  # [batch_size, seq_len, num_nodes, node_dim]

        # Create broadcasting indices using einops for clearer operations
        batch_indices = einops.repeat(
            torch.arange(batch_size, device=node_features.device),
            "batch_size -> batch_size seq_len num_edges",
            seq_len=1,
            num_edges=1,
        )
        seq_indices = einops.repeat(
            torch.arange(seq_len, device=node_features.device),
            "seq_len -> batch_size seq_len num_edges",
            batch_size=1,
            num_edges=1,
        )
        target_indices = einops.repeat(tgt_idx, "num_edges -> batch_size seq_len num_edges", batch_size=1, seq_len=1)

        # Vectorized aggregation using index_put with accumulate=True
        aggregated.index_put_((batch_indices, seq_indices, target_indices), messages, accumulate=True)

        # Add aggregated messages to current node features
        updated_sequence = node_features + aggregated

        return updated_sequence

    def forward(self, node_features, edge_index, edge_features=None, causal_mask=None):
        """
        Args:
            node_features: [batch_size, seq_len, num_nodes, node_dim] - sequence of node features
            edge_index: [2, num_edges] - source and target node indices
            edge_features: [num_edges, edge_dim] - edge features (static)
            causal_mask: [seq_len, seq_len] - optional precomputed causal mask

        Returns:
            updated_node_features: [batch_size, seq_len, num_nodes, node_dim]
        """
        # Perform message passing aggregation
        updated_sequence = self._message_passing(node_features, edge_index, edge_features)

        # Transform to node-centric view using einops for transformer processing
        node_centric = einops.rearrange(
            updated_sequence, "batch_size seq_len num_nodes node_dim -> batch_size num_nodes seq_len node_dim"
        )

        # Apply InContextTransformer that processes all nodes in parallel
        contextualized_node_centric = self.incontext_transformer(
            node_centric, src_mask=causal_mask
        )  # [batch_size, num_nodes, seq_len, node_dim]

        # Transform back to sequence-centric view using einops
        contextualized_features = einops.rearrange(
            contextualized_node_centric,
            "batch_size num_nodes seq_len node_dim -> batch_size seq_len num_nodes node_dim",
        )

        return contextualized_features


class GICONFrames_absPE(nn.Module):
    """
    Graph-based In-Context Operator Network with multi-frame support.
    This version handles multi-frame conditioning by concatenating timeframes on the feature axis.
    Adds absolute positional embeddings once before entering GICON blocks.
    """

    def __init__(
        self,
        node_wise_transformer: nn.Module,
        input_dim: int,
        node_dim: int = 256,
        edge_dim: int = 256,
        edge_attr_dim: int = 2,
        message_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        max_demo_num: int = 10,
        window_size: int = 24,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_demo_num = max_demo_num
        self.window_size = window_size

        # Separate projections for multi-frame cond (window_size*input_dim) and single-frame qoi (input_dim)
        self.pre_proj_cond = nn.Linear(
            in_features=window_size * input_dim, out_features=node_dim
        )  # For flattened multi-frame
        self.pre_proj_qoi = nn.Linear(in_features=input_dim, out_features=node_dim)  # For single-frame

        # Output projection back to single frame dimension
        self.post_proj = nn.Linear(in_features=node_dim, out_features=input_dim)

        # Edge projection layer (edge_attr_dim is fixed at 2)
        self.edge_proj = nn.Linear(in_features=edge_attr_dim, out_features=edge_dim)

        # Learnable positional embeddings similar to absolute PE
        # Example-level embeddings: one per example (demo examples + 1 for quest)
        self.example_embeddings = nn.Embedding(max_demo_num + 1, node_dim)
        # Token-type embeddings: 0 for cond, 1 for qoi
        self.type_embeddings = nn.Embedding(2, node_dim)

        # Initialize embeddings
        nn.init.trunc_normal_(self.example_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.type_embeddings.weight, std=0.02)

        # GICON blocks with per-node in-context learning
        # Each block gets its own independent copy of the transformer
        self.gicon_blocks = nn.ModuleList(
            [
                InContextMessagePassingLayer(node_dim, edge_dim, message_dim, copy.deepcopy(node_wise_transformer))
                for _ in range(num_layers)
            ]
        )

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create lower triangular causal mask dynamically based on sequence length.

        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on

        Returns:
            causal_mask: [seq_len, seq_len] mask with -inf for masked positions, 0 for valid positions
        """
        # Create lower triangular mask for causal attention
        # Use -inf for masked positions, 0 for unmasked positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float("-inf"))
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
        return causal_mask

    def _add_positional_embeddings(self, node_features):
        """
        Add learnable positional embeddings to node features.

        Args:
            node_features: [batch_size, seq_len, num_nodes, node_dim]

        Returns:
            node_features with positional embeddings added
        """
        batch_size, seq_len, num_nodes, node_dim = node_features.shape
        device = node_features.device

        # Create position indices for the sequence
        positions = torch.arange(seq_len, device=device)

        # Map sequence positions to example indices
        # For sequence [cond1, qoi1, cond2, qoi2, ..., cond_n, qoi_n, quest_cond]:
        # - Demo examples: positions 0-1 -> example 0, positions 2-3 -> example 1, etc.
        # - Quest: last position -> example max_demo_num
        example_indices = positions // 2  # [0, 0, 1, 1, 2, 2, ..., max_demo_num-1, max_demo_num-1, max_demo_num]

        # Map sequence positions to position types (cond/qoi)
        # Even positions (0, 2, 4, ...) are cond (type 0)
        # Odd positions (1, 3, 5, ...) are qoi (type 1)
        position_types = positions % 2

        # Get embeddings
        example_embs = self.example_embeddings(example_indices)  # [seq_len, node_dim]
        type_embs = self.type_embeddings(position_types)  # [seq_len, node_dim]

        # Combine embeddings
        total_embs = example_embs + type_embs  # [seq_len, node_dim]

        # Expand to match input shape
        total_embs = einops.repeat(
            total_embs, "seq_len node_dim -> batch num_nodes seq_len node_dim", batch=batch_size, num_nodes=num_nodes
        )

        # Rearrange to match node_features shape [batch_size, seq_len, num_nodes, node_dim]
        total_embs = einops.rearrange(
            total_embs, "batch num_nodes seq_len node_dim -> batch seq_len num_nodes node_dim"
        )

        return node_features + total_embs

    def forward(self, data, graph: Any, **kwargs):
        """Unified forward pass with v-only sequence and GNN message passing for multi-frame data.

        Args:
            data: Dictionary containing demo_cond_v, demo_qoi_v, quest_cond_v
                  - demo_cond_v: [batch_size, demo_num, window_size, num_node, dim]
                  - demo_qoi_v: [batch_size, demo_num, num_node, dim]
                  - quest_cond_v: [batch_size, 1, window_size, num_node, dim]
            graph: Graph object containing edge_index, edge_attr, node_attr
            **kwargs: Additional keyword arguments
        """
        demo_cond_v = data["demo_cond_v"]  # [batch_size, demo_num, window_size, num_node, dim]
        demo_qoi_v = data["demo_qoi_v"]  # [batch_size, demo_num, num_node, dim]
        quest_cond_v = data["quest_cond_v"]  # [batch_size, 1, window_size, num_node, dim]

        batch_size, demo_num, window_size, num_nodes, dim = demo_cond_v.shape

        # Extract graph structure
        device = demo_cond_v.device
        edge_index = torch.from_numpy(graph.edge_index).long().to(device)
        edge_attr = torch.from_numpy(graph.edge_attr).float().to(device)

        # Flatten multi-frame cond data: [batch, demo, window, nodes, dim] -> [batch, demo, nodes, window*dim]
        demo_cond_v_flat = einops.rearrange(demo_cond_v, "batch demo window nodes dim -> batch demo nodes (window dim)")
        quest_cond_v_flat = einops.rearrange(quest_cond_v, "batch one window nodes dim -> batch one nodes (window dim)")

        # Project separately: cond (multi-frame) and qoi (single-frame)
        demo_cond_proj = self.pre_proj_cond(demo_cond_v_flat)  # [batch, demo, nodes, node_dim]
        demo_qoi_proj = self.pre_proj_qoi(demo_qoi_v)  # [batch, demo, nodes, node_dim]
        quest_cond_proj = self.pre_proj_cond(quest_cond_v_flat)  # [batch, 1, nodes, node_dim]

        # Stack demo_cond and demo_qoi for interleaving
        demo_stack = torch.stack([demo_cond_proj, demo_qoi_proj], dim=2)  # [batch, demo, 2, nodes, node_dim]

        # Interleave demo conditions and QoIs
        demo_interleaved = einops.rearrange(
            demo_stack, "batch demo two nodes dim -> batch (demo two) nodes dim"
        )  # [batch, 2*demo_num, nodes, node_dim]

        # Concatenate with quest condition along sequence dimension
        node_features = torch.cat([demo_interleaved, quest_cond_proj], dim=1)  # [batch, 2*demo_num+1, nodes, node_dim]

        seq_len = node_features.shape[1]

        # Add learnable positional embeddings (example-level + type-level)
        node_features = self._add_positional_embeddings(node_features)

        edge_features = self.edge_proj(edge_attr)

        # Create causal mask once for all GICON blocks
        causal_mask = self._create_causal_mask(seq_len, device)

        # Apply GICON blocks for spatial message passing at each time step
        for gicon_block in self.gicon_blocks:
            # GICON expects [batch_size, seq_len, num_nodes, node_dim]
            node_features = gicon_block(node_features, edge_index, edge_features, causal_mask)

        # Apply final projection
        output = self.post_proj(node_features)  # [batch_size, seq_len, num_nodes, output_dim]

        # Extract demo predictions and quest prediction like VICON
        # Demo predictions: all demo cond positions (even indices from 0 to 2*demo_num-2)
        demo_indices = torch.arange(0, 2 * demo_num, 2)  # [0, 2, 4, ..., 2*demo_num-2]
        demo_pred_v = output[:, demo_indices, :, :]  # [batch_size, demo_num, num_nodes, full_features]

        # Quest prediction: the last position (2*demo_num)
        quest_pred_v = output[:, 2 * demo_num : 2 * demo_num + 1, :, :]  # [batch_size, 1, num_nodes, full_features]

        return {"demo_pred_v": demo_pred_v, "quest_pred_v": quest_pred_v}


class GICONFrames(nn.Module):
    """
    Graph-based In-Context Operator Network with multi-frame support.
    This version handles multi-frame conditioning by concatenating timeframes on the feature axis.
    Positional encoding is handled by the attention layers themselves (no absolute PE at model level).
    """

    def __init__(
        self,
        node_wise_transformer: nn.Module,
        input_dim: int,
        node_dim: int = 256,
        edge_dim: int = 256,
        edge_attr_dim: int = 2,
        message_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        window_size: int = 24,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size

        # Separate projections for multi-frame cond (window_size*input_dim) and single-frame qoi (input_dim)
        self.pre_proj_cond = nn.Linear(
            in_features=window_size * input_dim, out_features=node_dim
        )  # For flattened multi-frame
        self.pre_proj_qoi = nn.Linear(in_features=input_dim, out_features=node_dim)  # For single-frame

        # Output projection back to single frame dimension
        self.post_proj = nn.Linear(in_features=node_dim, out_features=input_dim)

        # Edge projection layer (edge_attr_dim is fixed at 2)
        self.edge_proj = nn.Linear(in_features=edge_attr_dim, out_features=edge_dim)

        # GICON blocks with per-node in-context learning
        # Each block gets its own independent copy of the transformer
        self.gicon_blocks = nn.ModuleList(
            [
                InContextMessagePassingLayer(node_dim, edge_dim, message_dim, copy.deepcopy(node_wise_transformer))
                for _ in range(num_layers)
            ]
        )

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create lower triangular causal mask dynamically based on sequence length.

        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on

        Returns:
            causal_mask: [seq_len, seq_len] mask with -inf for masked positions, 0 for valid positions
        """
        # Create lower triangular mask for causal attention
        # Use -inf for masked positions, 0 for unmasked positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float("-inf"))
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
        return causal_mask

    def forward(self, data, graph: Any, **kwargs):
        """Unified forward pass with v-only sequence and GNN message passing for multi-frame data.

        Args:
            data: Dictionary containing demo_cond_v, demo_qoi_v, quest_cond_v
                  - demo_cond_v: [batch_size, demo_num, window_size, num_node, dim]
                  - demo_qoi_v: [batch_size, demo_num, num_node, dim]
                  - quest_cond_v: [batch_size, 1, window_size, num_node, dim]
            graph: Graph object containing edge_index, edge_attr, node_attr
            **kwargs: Additional keyword arguments
        """
        demo_cond_v = data["demo_cond_v"]  # [batch_size, demo_num, window_size, num_node, dim]
        demo_qoi_v = data["demo_qoi_v"]  # [batch_size, demo_num, num_node, dim]
        quest_cond_v = data["quest_cond_v"]  # [batch_size, 1, window_size, num_node, dim]

        batch_size, demo_num, window_size, num_nodes, dim = demo_cond_v.shape

        # Extract graph structure
        device = demo_cond_v.device
        edge_index = torch.from_numpy(graph.edge_index).long().to(device)
        edge_attr = torch.from_numpy(graph.edge_attr).float().to(device)

        # Flatten multi-frame cond data: [batch, demo, window, nodes, dim] -> [batch, demo, nodes, window*dim]
        demo_cond_v_flat = einops.rearrange(demo_cond_v, "batch demo window nodes dim -> batch demo nodes (window dim)")
        quest_cond_v_flat = einops.rearrange(quest_cond_v, "batch one window nodes dim -> batch one nodes (window dim)")

        # Project separately: cond (multi-frame) and qoi (single-frame)
        demo_cond_proj = self.pre_proj_cond(demo_cond_v_flat)  # [batch, demo, nodes, node_dim]
        demo_qoi_proj = self.pre_proj_qoi(demo_qoi_v)  # [batch, demo, nodes, node_dim]
        quest_cond_proj = self.pre_proj_cond(quest_cond_v_flat)  # [batch, 1, nodes, node_dim]

        # Stack demo_cond and demo_qoi for interleaving
        demo_stack = torch.stack([demo_cond_proj, demo_qoi_proj], dim=2)  # [batch, demo, 2, nodes, node_dim]

        # Interleave demo conditions and QoIs
        demo_interleaved = einops.rearrange(
            demo_stack, "batch demo two nodes dim -> batch (demo two) nodes dim"
        )  # [batch, 2*demo_num, nodes, node_dim]

        # Concatenate with quest condition along sequence dimension
        node_features = torch.cat([demo_interleaved, quest_cond_proj], dim=1)  # [batch, 2*demo_num+1, nodes, node_dim]

        seq_len = node_features.shape[1]

        edge_features = self.edge_proj(edge_attr)

        # Create causal mask once for all GICON blocks
        causal_mask = self._create_causal_mask(seq_len, device)

        # Apply GICON blocks for spatial message passing at each time step
        # Note: Positional encoding is now handled within the attention layers
        for gicon_block in self.gicon_blocks:
            # GICON expects [batch_size, seq_len, num_nodes, node_dim]
            node_features = gicon_block(node_features, edge_index, edge_features, causal_mask)

        # Apply final projection
        output = self.post_proj(node_features)  # [batch_size, seq_len, num_nodes, output_dim]

        # Extract demo predictions and quest prediction like VICON
        # Demo predictions: all demo cond positions (even indices from 0 to 2*demo_num-2)
        demo_indices = torch.arange(0, 2 * demo_num, 2)  # [0, 2, 4, ..., 2*demo_num-2]
        demo_pred_v = output[:, demo_indices, :, :]  # [batch_size, demo_num, num_nodes, full_features]

        # Quest prediction: the last position (2*demo_num)
        quest_pred_v = output[:, 2 * demo_num : 2 * demo_num + 1, :, :]  # [batch_size, 1, num_nodes, full_features]

        return {"demo_pred_v": demo_pred_v, "quest_pred_v": quest_pred_v}
