import math

import einops
import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding


class RoleEncodingStrategy(nn.Module):
    """
    Base class for handling condition/QoI role encoding strategies.

    Provides two mechanisms for role differentiation:
    - Input-level: Modify token embeddings before attention
    - Logit-level: Add bias to attention scores
    """

    def process_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply input-level role encoding to hidden states.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, embed_dim]

        Returns:
            Modified hidden states with same shape
        """
        return hidden_states

    def build_bias(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        num_nodes: int,
        num_heads: int,
    ) -> torch.Tensor | None:
        """
        Build attention bias for logit-level role encoding.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, embed_dim]
            batch_size: Batch size
            num_nodes: Number of nodes in graph
            num_heads: Number of attention heads

        Returns:
            Attention bias [(batch*num_nodes), num_heads, seq_len, seq_len] or None
        """
        return None


class NoRoleEncoding(RoleEncodingStrategy):
    """No role encoding applied - condition and QoI tokens treated identically."""

    pass


class InputRoleEncoding(RoleEncodingStrategy):
    """
    Input-level role encoding using learnable offset vectors.

    Adds +r to condition tokens and -r to QoI tokens, where r is a learnable vector.
    This creates an input-space separation between the two token types.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Learnable role vector for cond/qoi differentiation
        self.role_vec = nn.Parameter(torch.randn(embed_dim))

    def process_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Add ±role_vec offset to condition/QoI tokens.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, embed_dim]

        Returns:
            Modified hidden states with role offsets applied
        """
        batch_size, num_nodes, seq_len, embed_dim = hidden_states.shape
        device = hidden_states.device

        # Create position-based masks for cond (even) and qoi (odd) tokens
        positions = torch.arange(seq_len, device=device)
        cond_mask = positions % 2 == 0  # [seq_len]
        qoi_mask = ~cond_mask  # [seq_len]

        # Expand masks for broadcasting
        cond_mask = einops.repeat(
            cond_mask,
            "seq_len -> batch num_nodes seq_len embed_dim",
            batch=batch_size,
            num_nodes=num_nodes,
            embed_dim=embed_dim,
        )
        qoi_mask = einops.repeat(
            qoi_mask,
            "seq_len -> batch num_nodes seq_len embed_dim",
            batch=batch_size,
            num_nodes=num_nodes,
            embed_dim=embed_dim,
        )

        # Expand role vector for broadcasting
        role_vec = einops.repeat(
            self.role_vec,
            "embed_dim -> batch num_nodes seq_len embed_dim",
            batch=batch_size,
            num_nodes=num_nodes,
            seq_len=seq_len,
        )

        # Apply ±role_vec based on token type
        hidden_states = hidden_states + cond_mask * role_vec - qoi_mask * role_vec

        return hidden_states


class LogitRoleEncoding(RoleEncodingStrategy):
    """
    Logit-level role encoding using learnable attention bias.

    Adds per-head bias values for 4 interaction types:
    - 0: cond seeing cond
    - 1: cond seeing qoi
    - 2: qoi seeing cond
    - 3: qoi seeing qoi
    """

    def __init__(self, num_heads: int):
        super().__init__()
        # Learnable bias for 4 cond/qoi interaction types
        self.bias = nn.Embedding(4, num_heads)

    def build_bias(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        num_nodes: int,
        num_heads: int,
    ) -> torch.Tensor:
        """
        Build attention bias matrix for cond/qoi interactions.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, embed_dim]
            batch_size: Batch size
            num_nodes: Number of nodes in graph
            num_heads: Number of attention heads

        Returns:
            Attention bias [(batch*num_nodes), num_heads, seq_len, seq_len]
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        seq_len = hidden_states.shape[2]

        # Create position grids for all token pairs
        positions = torch.arange(seq_len, device=device)
        i_grid, j_grid = torch.meshgrid(positions, positions, indexing="ij")

        # Determine token types (even=cond, odd=qoi)
        i_is_qoi = (i_grid % 2) == 1
        j_is_qoi = (j_grid % 2) == 1

        # Map to interaction type indices
        # 0: cond→cond, 1: cond→qoi, 2: qoi→cond, 3: qoi→qoi
        indices = (i_is_qoi.long() * 2) + j_is_qoi.long()

        # Get bias values and reshape to [num_heads, seq_len, seq_len]
        head_bias = self.bias(indices)  # [seq_len, seq_len, num_heads]
        head_bias = einops.rearrange(head_bias, "s1 s2 h -> h s1 s2")
        head_bias = head_bias.to(dtype)

        # Expand to match batch and node dimensions
        head_bias = einops.repeat(
            head_bias, "h s1 s2 -> (batch num_nodes) h s1 s2", batch=batch_size, num_nodes=num_nodes
        )

        return head_bias


def build_role_encoding(mode: str, embed_dim: int, num_heads: int) -> RoleEncodingStrategy:
    """
    Factory function to create role encoding strategy based on mode.

    Args:
        mode: Encoding mode ("none", "input", or "logit")
        embed_dim: Embedding dimension (for input mode)
        num_heads: Number of attention heads (for logit mode)

    Returns:
        RoleEncodingStrategy instance
    """
    mode = (mode or "none").lower()
    if mode == "input":
        return InputRoleEncoding(embed_dim)
    if mode == "logit":
        return LogitRoleEncoding(num_heads)
    return NoRoleEncoding()


class BaseMultiheadAttention(nn.Module):
    """
    Base multihead attention using LlamaAttention as the core component.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Optional pre-normalization (disabled; outer transformer handles it)
        self.pre_norm = nn.LayerNorm(embed_dim)

        self.role_encoding = NoRoleEncoding()

        # Create LlamaConfig for the attention layer
        config = LlamaConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            attention_dropout=dropout,
            _attn_implementation="eager",
        )

        # Create LlamaAttention layer
        self.attention = LlamaAttention(config, layer_idx=0)

        # Create RoPE for generating position embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Base forward pass with LlamaAttention.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [batch_size, num_heads, seq_len, seq_len] causal mask for attention
            position_ids: [batch_size, seq_len] position indices (can be float!)

        Returns:
            attn_output: [batch_size, num_nodes, seq_len, node_dim]
        """
        batch_size, num_nodes, seq_len, node_dim = hidden_states.shape

        if self.pre_norm is not None:
            hidden_states = self.pre_norm(hidden_states)

        hidden_states = self.role_encoding.process_input(hidden_states)

        # Reshape to [batch_size * num_nodes, seq_len, node_dim] for LlamaAttention
        hidden_states_flat = einops.rearrange(
            hidden_states, "batch num_nodes seq_len node_dim -> (batch num_nodes) seq_len node_dim"
        )

        # Expand position_ids to match flattened batch dimension
        if position_ids is not None:
            position_ids_expanded = einops.repeat(
                position_ids, "batch seq_len -> (batch num_nodes) seq_len", num_nodes=num_nodes
            )
        else:
            position_ids_expanded = None

        # Expand mask to match flattened batch dimension
        mask_expanded = None
        if mask is not None:
            mask_expanded = einops.repeat(
                mask,
                "batch num_heads seq_len1 seq_len2 -> (batch num_nodes) num_heads seq_len1 seq_len2",
                num_nodes=num_nodes,
            )

        role_bias = self.role_encoding.build_bias(hidden_states, batch_size, num_nodes, self.num_heads)
        if role_bias is not None:
            mask_expanded = role_bias if mask_expanded is None else mask_expanded + role_bias

        # Get position embeddings (cos, sin) from RoPE
        cos, sin = self.rotary_emb(hidden_states_flat, position_ids_expanded)
        position_embeddings = (cos, sin)

        # Call LlamaAttention
        attn_output_flat, attn_weights_flat = self.attention(
            hidden_states=hidden_states_flat,
            position_embeddings=position_embeddings,
            attention_mask=mask_expanded,
        )

        # Reshape output back to [batch_size, num_nodes, seq_len, node_dim]
        attn_output = einops.rearrange(
            attn_output_flat,
            "(batch num_nodes) seq_len node_dim -> batch num_nodes seq_len node_dim",
            batch=batch_size,
            num_nodes=num_nodes,
        )

        return attn_output


class VanillaAttentionLayer(BaseMultiheadAttention):
    """
    Attention layer specified for GICON_absPE.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with RoPE.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] optional causal mask
        """
        batch_size, num_nodes, seq_len, _ = hidden_states.shape

        # Use zero position_ids
        position_ids = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.long)

        # Expand mask to match batch size
        if mask is not None:
            mask = einops.repeat(
                mask,
                "seq_len1 seq_len2 -> batch num_heads seq_len1 seq_len2",
                batch=batch_size,
                num_heads=self.num_heads,
            )

        return super().forward(hidden_states, mask, position_ids)


class RoPEAttentionLayer(BaseMultiheadAttention):
    """
    Attention layer with Rotary Positional Embedding (RoPE).
    Uses position_ids to apply rotary embeddings.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with RoPE.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] optional causal mask
        """
        batch_size, num_nodes, seq_len, _ = hidden_states.shape

        # Create normal position_ids for RoPE: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
        position_ids = einops.repeat(position_ids, "seq_len -> batch seq_len", batch=batch_size)

        # Expand mask to match batch size
        if mask is not None:
            mask = einops.repeat(
                mask,
                "seq_len1 seq_len2 -> batch num_heads seq_len1 seq_len2",
                batch=batch_size,
                num_heads=self.num_heads,
            )

        return super().forward(hidden_states, mask, position_ids)


class SWinAttentionLayer(BaseMultiheadAttention):
    """
    Shifted Window Attention layer with learnable positional bias.
    Uses learnable embeddings for relative positions instead of RoPE.
    """

    def __init__(self, embed_dim: int, num_heads: int, max_demo_num: int = 10, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout)

        # Learnable positional bias using relative positions
        # Stores bias for relative distances from -2 * max_demo_num to +2 * max_demo_num
        self.max_demo_num = max_demo_num
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.pos_bias = nn.Embedding(4 * max_demo_num + 1, num_heads)
        nn.init.trunc_normal_(self.pos_bias.weight, std=0.02)

    def _create_positional_bias_matrix(self, seq_len, device):
        """Create learnable positional bias matrix using relative positions."""
        indices = torch.zeros(seq_len, seq_len, device=device, dtype=torch.long)

        head_dim = self.embed_dim // self.num_heads

        # Create relative position matrix using broadcasting
        i_positions = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        j_positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        relative_positions = j_positions - i_positions  # [seq_len, seq_len]

        # Clamp and convert to embedding indices
        indices = (
            torch.clamp(relative_positions, -2 * self.max_demo_num, 2 * self.max_demo_num) + seq_len - 1
        )  # seq_len is no larger than 2 * max_demo_num + 1

        # Get bias values from embedding: [seq_len, seq_len, num_heads]
        head_bias = self.pos_bias(indices)

        # Reshape to [num_heads, seq_len, seq_len]
        pos_bias_matrix = head_bias.permute(2, 0, 1) / math.sqrt(head_dim)

        return pos_bias_matrix

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with learnable positional bias (no RoPE).

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] optional causal mask
        """
        batch_size, num_nodes, seq_len, _ = hidden_states.shape

        # Create positional bias matrix
        pos_bias_matrix = self._create_positional_bias_matrix(seq_len, hidden_states.device)
        # pos_bias_matrix = 4 * torch.sigmoid(pos_bias_matrix)  # Scale bias values

        # Expand to match batch size [batch_size, num_heads, seq_len, seq_len]
        pos_bias_expanded = einops.repeat(
            pos_bias_matrix, "num_heads seq_len1 seq_len2 -> batch num_heads seq_len1 seq_len2", batch=batch_size
        )

        # Combine with causal mask if provided
        combined_mask = mask + pos_bias_expanded if mask is not None else pos_bias_expanded

        # Use zero position_ids (not used when we have manual bias)
        position_ids = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.long)

        return super().forward(hidden_states, combined_mask, position_ids)


class AttnBiasAttentionLayer(BaseMultiheadAttention):
    """
    Attention layer with content-aware attention bias.
    Supports different similarity computation modes for generating bias.

    Condition embedding modes:
        - None: Simple mean pooling over nodes
        - "pooling": Learnable MLP with spatial attention pooling
        - "learning": Mean pooling over nodes followed by learnable MLP transformation

    Role modes (via build_role_encoding):
        - None: No role differentiation
        - "input": Add ±role_vec to cond/qoi tokens
        - "logit": Add learnable bias for cond/qoi interaction types
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias_mode: str = "prod",  # "prod" or "attn"
        cond_embedding_mode: str = None,  # None, "pooling", "learning"
        role_mode: str = None,  # None, "input", "logit"
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.bias_mode = bias_mode
        self.cond_embedding_mode = cond_embedding_mode

        # Initialize role encoding strategy
        self.role_encoding = build_role_encoding(role_mode, embed_dim, num_heads)

        # Head-specific projections for learned similarity - always initialize to ensure consistent parameter count
        self.cond_q_proj = nn.Linear(embed_dim, embed_dim, bias=False) if bias_mode == "attn" else None
        self.cond_k_proj = nn.Linear(embed_dim, embed_dim, bias=False) if bias_mode == "attn" else None

        # Condition embedding projections - always initialize to ensure consistent parameter count
        if cond_embedding_mode == "pooling":
            self.cond_embedding_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, embed_dim), nn.Dropout(0.1)
            )
            self.spatial_attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4), nn.ReLU(), nn.Linear(embed_dim // 4, 1), nn.Sigmoid()
            )
        elif cond_embedding_mode == "learning":
            # MLP to map averaged condition features to learned embeddings
            self.cond_embedding_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, embed_dim), nn.Dropout(0.1)
            )
        else:
            self.cond_embedding_proj = None
            self.spatial_attention = None

    def _compute_cond_similarities_prod(self, cond_embeddings) -> torch.Tensor:
        """Compute similarities using group inner product."""
        batch_size, num_conds, embed_dim = cond_embeddings.shape
        group_size = embed_dim // self.num_heads

        # Reshape to [batch_size, num_conds, n_heads, group_size]
        grouped_embeddings = einops.rearrange(
            cond_embeddings,
            "batch num_conds (n_heads group_size) -> batch num_conds n_heads group_size",
            n_heads=self.num_heads,
        )

        # Compute similarities
        pattern = "batch num_conds1 n_heads group_size, batch num_conds2 n_heads group_size"
        pattern += " -> batch n_heads num_conds1 num_conds2"
        similarities = einops.einsum(grouped_embeddings, grouped_embeddings, pattern)

        # Normalize by group size
        similarities = similarities / (group_size**0.5)

        return similarities

    def _compute_cond_similarities_learned(self, cond_embeddings) -> torch.Tensor:
        """Compute similarities using learned projections."""
        batch_size, num_conds, embed_dim = cond_embeddings.shape
        proj_dim = embed_dim // self.num_heads

        # Project to Q and K
        assert self.cond_q_proj is not None and self.cond_k_proj is not None, (
            "Projections not initialized for attn mode"
        )
        q = self.cond_q_proj(cond_embeddings)
        k = self.cond_k_proj(cond_embeddings)

        # Reshape to [batch_size, num_conds, n_heads, proj_dim]
        q = einops.rearrange(
            q, "batch num_conds (n_heads proj_dim) -> batch num_conds n_heads proj_dim", n_heads=self.num_heads
        )
        k = einops.rearrange(
            k, "batch num_conds (n_heads proj_dim) -> batch num_conds n_heads proj_dim", n_heads=self.num_heads
        )

        # Compute attention scores
        pattern = "batch num_conds1 n_heads proj_dim, batch num_conds2 n_heads proj_dim"
        pattern += " -> batch n_heads num_conds1 num_conds2"
        similarities = einops.einsum(q, k, pattern)

        # Scale by sqrt(proj_dim)
        similarities = similarities / (proj_dim**0.5)

        return similarities

    def _generate_attention_bias(self, node_features) -> torch.Tensor:
        """
        Generate content-aware attention bias based on condition similarities.

        Args:
            node_features: [batch_size, num_nodes, seq_len, embed_dim]

        Returns:
            attn_bias: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, num_nodes, seq_len, embed_dim = node_features.shape

        # Extract condition embeddings (even indices)
        cond_indices = torch.arange(0, seq_len, 2)
        cond_features = node_features[:, :, cond_indices, :]
        batch_size, num_nodes, num_conds, embed_dim = cond_features.shape

        if self.cond_embedding_mode == "pooling":
            # Use learnable embedding with pooling
            assert self.cond_embedding_proj is not None and self.spatial_attention is not None, (
                "Embedding layers not initialized for pooling mode"
            )
            cond_features_flat = einops.rearrange(
                cond_features, "batch num_nodes num_conds embed_dim -> (batch num_nodes num_conds) embed_dim"
            )
            projected_features = self.cond_embedding_proj(cond_features_flat)
            projected_features = einops.rearrange(
                projected_features,
                "(batch num_nodes num_conds) embed_dim -> batch num_nodes num_conds embed_dim",
                batch=batch_size,
                num_nodes=num_nodes,
                num_conds=num_conds,
            )
            attention_weights = self.spatial_attention(projected_features)
            cond_embeddings = einops.reduce(
                projected_features * attention_weights,
                "batch num_nodes num_conds embed_dim -> batch num_conds embed_dim",
                "sum",
            )
        elif self.cond_embedding_mode == "learning":
            # Average over nodes, then apply MLP to learn embeddings
            assert self.cond_embedding_proj is not None, "Embedding layer not initialized for learning mode"
            # Average over nodes: [batch num_nodes num_conds embed_dim] -> [batch num_conds embed_dim]
            cond_embeddings = einops.reduce(
                cond_features, "batch num_nodes num_conds embed_dim -> batch num_conds embed_dim", "mean"
            )
            # Apply MLP: [batch num_conds embed_dim] -> [batch num_conds embed_dim]
            cond_embeddings_flat = einops.rearrange(
                cond_embeddings, "batch num_conds embed_dim -> (batch num_conds) embed_dim"
            )
            cond_embeddings_flat = self.cond_embedding_proj(cond_embeddings_flat)
            cond_embeddings = einops.rearrange(
                cond_embeddings_flat,
                "(batch num_conds) embed_dim -> batch num_conds embed_dim",
                batch=batch_size,
                num_conds=num_conds,
            )
        else:
            # Simple mean pooling over nodes
            cond_embeddings = einops.reduce(
                cond_features, "batch num_nodes num_conds embed_dim -> batch num_conds embed_dim", "mean"
            )

        # Compute similarities between conditions
        if self.bias_mode == "attn":
            cond_similarities = self._compute_cond_similarities_learned(cond_embeddings)
        elif self.bias_mode == "prod":
            cond_similarities = self._compute_cond_similarities_prod(cond_embeddings)
        else:
            raise ValueError(f"Unknown bias mode: {self.bias_mode}")

        # Create sequence-level bias matrix
        attn_bias = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=node_features.device)

        # Fill the bias matrix using vectorized operations
        # Create condition index mappings for all positions
        i_indices = torch.arange(seq_len, device=attn_bias.device) // 2  # [seq_len]
        j_indices = torch.arange(seq_len, device=attn_bias.device) // 2  # [seq_len]

        # Create meshgrid for all position pairs
        cond_i_grid, cond_j_grid = torch.meshgrid(i_indices, j_indices, indexing="ij")  # [seq_len, seq_len]

        # Create mask for valid condition indices
        valid_mask = (cond_i_grid < cond_similarities.shape[2]) & (cond_j_grid < cond_similarities.shape[2])

        # Use advanced indexing to fill the bias matrix
        attn_bias[:, :, valid_mask] = cond_similarities[:, :, cond_i_grid[valid_mask], cond_j_grid[valid_mask]]

        return attn_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with content-aware attention bias.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] optional causal mask
        """
        # Generate content-aware bias
        attn_bias = self._generate_attention_bias(hidden_states)

        # Combine with causal mask if provided
        combined_mask = mask + attn_bias if mask is not None else attn_bias

        # Use zero position_ids (bias handles positioning)
        batch_size, num_nodes, seq_len, _ = hidden_states.shape
        position_ids = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.long)

        return super().forward(hidden_states, combined_mask, position_ids)


class AttnBiasAttentionLayer_sanity(BaseMultiheadAttention):
    """
    Sanity check version of AttnBiasAttentionLayer using positionwise embeddings.
    Uses only position indices instead of actual content features for generating bias.

    This class uses max_demo_num parameter for defining the position embedding table size.

    Role modes (via build_role_encoding):
        - None: No role differentiation
        - "input": Add ±role_vec to cond/qoi tokens
        - "logit": Add learnable bias for cond/qoi interaction types
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias_mode: str = "prod",  # "prod" or "attn"
        max_demo_num: int = 10,
        role_mode: str = None,  # None, "input", "logit"
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.bias_mode = bias_mode
        self.max_demo_num = max_demo_num

        # Initialize role encoding strategy
        self.role_encoding = build_role_encoding(role_mode, embed_dim, num_heads)

        # Head-specific projections for learned similarity
        self.cond_q_proj = nn.Linear(embed_dim, embed_dim, bias=False) if bias_mode == "attn" else None
        self.cond_k_proj = nn.Linear(embed_dim, embed_dim, bias=False) if bias_mode == "attn" else None

        # Positional embeddings for sanity check
        self.cond_embedding_proj = nn.Embedding(max_demo_num + 1, embed_dim)

    def _compute_cond_similarities_prod(self, cond_embeddings) -> torch.Tensor:
        """Compute similarities using group inner product."""
        batch_size, num_conds, embed_dim = cond_embeddings.shape
        group_size = embed_dim // self.num_heads

        # Reshape to [batch_size, num_conds, n_heads, group_size]
        grouped_embeddings = einops.rearrange(
            cond_embeddings,
            "batch num_conds (n_heads group_size) -> batch num_conds n_heads group_size",
            n_heads=self.num_heads,
        )

        # Compute similarities
        pattern = "batch num_conds1 n_heads group_size, batch num_conds2 n_heads group_size"
        pattern += " -> batch n_heads num_conds1 num_conds2"
        similarities = einops.einsum(grouped_embeddings, grouped_embeddings, pattern)

        # Normalize by group size
        similarities = similarities / (group_size**0.5)

        return similarities

    def _compute_cond_similarities_learned(self, cond_embeddings) -> torch.Tensor:
        """Compute similarities using learned projections."""
        batch_size, num_conds, embed_dim = cond_embeddings.shape
        proj_dim = embed_dim // self.num_heads

        # Project to Q and K
        assert self.cond_q_proj is not None and self.cond_k_proj is not None, (
            "Projections not initialized for attn mode"
        )
        q = self.cond_q_proj(cond_embeddings)
        k = self.cond_k_proj(cond_embeddings)

        # Reshape to [batch_size, num_conds, n_heads, proj_dim]
        q = einops.rearrange(
            q, "batch num_conds (n_heads proj_dim) -> batch num_conds n_heads proj_dim", n_heads=self.num_heads
        )
        k = einops.rearrange(
            k, "batch num_conds (n_heads proj_dim) -> batch num_conds n_heads proj_dim", n_heads=self.num_heads
        )

        # Compute attention scores
        pattern = "batch num_conds1 n_heads proj_dim, batch num_conds2 n_heads proj_dim"
        pattern += " -> batch n_heads num_conds1 num_conds2"
        similarities = einops.einsum(q, k, pattern)

        # Scale by sqrt(proj_dim)
        similarities = similarities / (proj_dim**0.5)

        return similarities

    def _generate_attention_bias(self, node_features) -> torch.Tensor:
        """
        Generate content-aware attention bias based on positional embeddings (sanity check).

        Args:
            node_features: [batch_size, num_nodes, seq_len, embed_dim]

        Returns:
            attn_bias: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, num_nodes, seq_len, embed_dim = node_features.shape

        # Extract condition embeddings (even indices)
        cond_indices = torch.arange(0, seq_len, 2)
        cond_features = node_features[:, :, cond_indices, :]
        batch_size, num_nodes, num_conds, embed_dim = cond_features.shape

        # Use positional embeddings (sanity check - ignores actual content)
        position_indices = torch.arange(num_conds, device=cond_features.device, dtype=torch.long)
        cond_embeddings = self.cond_embedding_proj(position_indices)
        cond_embeddings = einops.repeat(
            cond_embeddings, "num_conds embed_dim -> batch num_conds embed_dim", batch=batch_size
        )

        # Compute similarities between conditions
        if self.bias_mode == "attn":
            cond_similarities = self._compute_cond_similarities_learned(cond_embeddings)
        elif self.bias_mode == "prod":
            cond_similarities = self._compute_cond_similarities_prod(cond_embeddings)
        else:
            raise ValueError(f"Unknown bias mode: {self.bias_mode}")

        # Create sequence-level bias matrix
        attn_bias = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=node_features.device)

        # Fill the bias matrix using vectorized operations
        i_indices = torch.arange(seq_len, device=attn_bias.device) // 2
        j_indices = torch.arange(seq_len, device=attn_bias.device) // 2

        # Create meshgrid for all position pairs
        cond_i_grid, cond_j_grid = torch.meshgrid(i_indices, j_indices, indexing="ij")

        # Create mask for valid condition indices
        valid_mask = (cond_i_grid < cond_similarities.shape[2]) & (cond_j_grid < cond_similarities.shape[2])

        # Use advanced indexing to fill the bias matrix
        attn_bias[:, :, valid_mask] = cond_similarities[:, :, cond_i_grid[valid_mask], cond_j_grid[valid_mask]]

        return attn_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with positional attention bias (sanity check).

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] optional causal mask
        """
        # Generate positional bias (sanity check)
        attn_bias = self._generate_attention_bias(hidden_states)

        # Combine with causal mask if provided
        combined_mask = mask + attn_bias if mask is not None else attn_bias

        # Use zero position_ids (bias handles positioning)
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[2]
        position_ids = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.long)

        return super().forward(hidden_states, combined_mask, position_ids)


class AttnWRoleBiasAttentionLayer(AttnBiasAttentionLayer):
    """
    Attention layer with both content-aware attention bias and learnable positional bias.
    Extends AttnBiasAttentionLayer by setting role_mode="logit" for cond/qoi interaction bias.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias_mode: str = "prod",  # "prod" or "attn"
        cond_embedding_mode: str = None,  # None, "pooling", "learning"
        dropout: float = 0.0,
    ):
        # Initialize parent with logit-level role encoding for positional bias
        super().__init__(embed_dim, num_heads, bias_mode, cond_embedding_mode, role_mode="logit", dropout=dropout)


class AttnWRoleBiasAttentionLayer_sanity(AttnBiasAttentionLayer_sanity):
    """
    Sanity check version combining positional condition embeddings with learnable positional bias.
    Extends AttnBiasAttentionLayer_sanity by setting role_mode="logit" for cond/qoi interaction bias.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias_mode: str = "prod",  # "prod" or "attn"
        max_demo_num: int = 10,
        dropout: float = 0.0,
    ):
        # Initialize parent with logit-level role encoding for positional bias
        super().__init__(embed_dim, num_heads, bias_mode, max_demo_num, role_mode="logit", dropout=dropout)


class AbsPEAttentionLayer(BaseMultiheadAttention):
    """
    Attention layer with absolute positional embeddings added before each attention.
    Similar to original GICON implementation.
    This provides an alternative to relative positional bias methods.
    """

    def __init__(self, embed_dim: int, num_heads: int, max_demo_num: int = 10, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout)
        self.max_demo_num = max_demo_num
        self.embed_dim = embed_dim

        # Learnable positional embeddings similar to absolute PE
        # Example-level embeddings: one per example (demo examples + 1 for quest)
        self.example_embeddings = nn.Embedding(max_demo_num + 1, embed_dim)
        # Token-type embeddings: 0 for cond, 1 for qoi
        self.type_embeddings = nn.Embedding(2, embed_dim)

        # Initialize embeddings
        nn.init.trunc_normal_(self.example_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.type_embeddings.weight, std=0.02)

    def _add_positional_embeddings(self, hidden_states):
        """
        Add learnable positional embeddings to hidden states.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, embed_dim]

        Returns:
            hidden_states with positional embeddings added
        """
        batch_size, num_nodes, seq_len, embed_dim = hidden_states.shape
        device = hidden_states.device

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
        example_embs = self.example_embeddings(example_indices)  # [seq_len, embed_dim]
        type_embs = self.type_embeddings(position_types)  # [seq_len, embed_dim]

        # Combine embeddings
        total_embs = example_embs + type_embs  # [seq_len, embed_dim]

        # Expand to match input shape
        total_embs = einops.repeat(
            total_embs, "seq_len embed_dim -> batch num_nodes seq_len embed_dim", batch=batch_size, num_nodes=num_nodes
        )

        return hidden_states + total_embs

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with absolute positional embeddings.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] optional causal mask
        """
        batch_size, num_nodes, seq_len, _ = hidden_states.shape

        # Add absolute positional embeddings
        hidden_states = self._add_positional_embeddings(hidden_states)

        # Expand mask to match batch size
        if mask is not None:
            mask = einops.repeat(
                mask,
                "seq_len1 seq_len2 -> batch num_heads seq_len1 seq_len2",
                batch=batch_size,
                num_heads=self.num_heads,
            )

        # Use zero position_ids (PE already added to hidden states)
        position_ids = torch.zeros((batch_size, seq_len), device=hidden_states.device, dtype=torch.long)

        return super().forward(hidden_states, mask, position_ids)


class HeadSetRoPEAttentionLayer(nn.Module):
    """
    Attention layer with head-wise Set-RoPE positional encoding using LlamaAttention.

    Set-RoPE approach:
    - Pools condition tokens (even indices) per example to create example-level embeddings
    - Generates head-wise rotation angles via linear readout from pooled embeddings
    - Applies zero-mean normalization across examples and tanh-bounded scaling
    - Uses custom RoPE embeddings with per-head angles

    Parallel implementation to BaseMultiheadAttention, also using LlamaAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        phi_max: float = math.pi,
        dropout: float = 0.0,
        role_mode: str = "none",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.phi_max = phi_max

        # Optional pre-norm (enabled in addition to outer norm)
        self.pre_norm = nn.LayerNorm(embed_dim)

        self.role_encoding = build_role_encoding(role_mode, embed_dim, num_heads)

        config = LlamaConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            attention_dropout=dropout,
            _attn_implementation="eager",
        )

        # Create LlamaAttention layer
        self.attention = LlamaAttention(config, layer_idx=0)

        # Head-wise angle readout for Set-RoPE
        self.phi_readout = nn.Linear(embed_dim, num_heads, bias=False)

    def _apply_rotary_emb(
        self, q: torch.Tensor, k: torch.Tensor, phi_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply custom rotary embeddings with per-head angles.

        Args:
            q: [batch, seq_len, num_heads, head_dim] query tensor
            k: [batch, seq_len, num_heads, head_dim] key tensor
            phi_tokens: [batch, seq_len, num_heads] rotation angles

        Returns:
            Tuple of rotated (q, k) tensors
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        assert head_dim % 2 == 0, "Head dimension must be even for rotary embeddings"

        # Reshape angles for broadcasting: [batch, seq_len, num_heads, 1]
        angles = einops.rearrange(phi_tokens, "batch seq_len num_heads -> batch seq_len num_heads 1")

        # Compute rotation components
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Split features into pairs and apply rotation
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

        return q_rotated, k_rotated

    def _compute_angles(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation angles for each position based on condition embeddings.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, embed_dim]

        Returns:
            phi_tokens: [batch_size, seq_len, num_heads] rotation angles
        """
        batch_size, num_nodes, seq_len, embed_dim = hidden_states.shape
        device = hidden_states.device

        # Map sequence positions to example IDs
        positions = torch.arange(seq_len, device=device)
        example_ids = positions // 2  # [0, 0, 1, 1, 2, 2, ...]
        cond_mask = positions % 2 == 0  # Even positions are conditions

        # Extract condition tokens and pool over nodes
        cond_hidden = hidden_states[:, :, cond_mask, :]  # [batch, num_nodes, num_examples, embed_dim]
        cond_embeddings = einops.reduce(
            cond_hidden, "batch num_nodes num_examples embed_dim -> batch num_examples embed_dim", "mean"
        )

        # Generate head-wise angles from condition embeddings
        phi_raw = self.phi_readout(cond_embeddings)  # [batch, num_examples, num_heads]

        # Apply zero-mean normalization and tanh-bounded scaling
        phi_raw = phi_raw - phi_raw.mean(dim=1, keepdim=True)
        phi = self.phi_max * torch.tanh(phi_raw)

        # Broadcast angles from examples to tokens
        example_ids_expanded = einops.repeat(
            example_ids, "seq_len -> batch seq_len num_heads", batch=batch_size, num_heads=self.num_heads
        )
        phi_tokens = torch.gather(phi, 1, example_ids_expanded)

        return phi_tokens  # [batch, seq_len, num_heads]

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass with Set-RoPE positional encoding.

        Args:
            hidden_states: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] optional causal mask

        Returns:
            attn_output: [batch_size, num_nodes, seq_len, node_dim]
        """
        batch_size, num_nodes, seq_len, node_dim = hidden_states.shape

        if self.pre_norm is not None:
            hidden_states = self.pre_norm(hidden_states)

        hidden_states = self.role_encoding.process_input(hidden_states)

        # Compute rotation angles from condition embeddings
        phi_tokens = self._compute_angles(hidden_states)  # [batch, seq_len, num_heads]

        # Reshape to [batch_size * num_nodes, seq_len, node_dim] for LlamaAttention
        hidden_states_flat = einops.rearrange(
            hidden_states, "batch num_nodes seq_len node_dim -> (batch num_nodes) seq_len node_dim"
        )

        # Broadcast angles to flattened batch dimension
        phi_tokens_flat = einops.repeat(
            phi_tokens, "batch seq_len num_heads -> (batch num_nodes) seq_len num_heads", num_nodes=num_nodes
        )

        # Expand mask to match flattened batch dimension
        mask_expanded = None
        if mask is not None:
            mask_expanded = einops.repeat(
                mask,
                "seq_len1 seq_len2 -> (batch num_nodes) num_heads seq_len1 seq_len2",
                batch=batch_size,
                num_nodes=num_nodes,
                num_heads=self.num_heads,
            )

        role_bias = self.role_encoding.build_bias(hidden_states, batch_size, num_nodes, self.num_heads)
        if role_bias is not None:
            mask_expanded = role_bias if mask_expanded is None else mask_expanded + role_bias

        # Get Q, K, V from LlamaAttention's projections
        q = self.attention.q_proj(hidden_states_flat)
        k = self.attention.k_proj(hidden_states_flat)
        v = self.attention.v_proj(hidden_states_flat)

        # Reshape for multi-head attention
        q = einops.rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = einops.rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = einops.rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        # Apply custom rotary embeddings with per-head angles
        q, k = self._apply_rotary_emb(q, k, phi_tokens_flat)

        # Reshape for attention computation
        q = einops.rearrange(q, "b s h d -> b h s d")
        k = einops.rearrange(k, "b s h d -> b h s d")
        v = einops.rearrange(v, "b s h d -> b h s d")

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k) / scale

        # Apply mask if provided
        if mask_expanded is not None:
            attn_scores = attn_scores + mask_expanded

        # Apply softmax and dropout
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        if self.training and self.attention.config.attention_dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention.config.attention_dropout)

        # Apply attention to values
        attn_output = torch.einsum("bhij,bhjd->bhid", attn_weights, v)

        # Concatenate heads and apply output projection
        attn_output = einops.rearrange(attn_output, "b h s d -> b s (h d)")
        attn_output = self.attention.o_proj(attn_output)

        # Reshape back to [batch_size, num_nodes, seq_len, node_dim]
        attn_output = einops.rearrange(
            attn_output,
            "(batch num_nodes) seq_len node_dim -> batch num_nodes seq_len node_dim",
            batch=batch_size,
            num_nodes=num_nodes,
        )

        return attn_output


class HeadSetRoPEWRoleBiasAttentionLayer(HeadSetRoPEAttentionLayer):
    """Convenience subclass that enables logit-level role bias."""

    def __init__(self, embed_dim: int, num_heads: int, phi_max: float = math.pi, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, phi_max, dropout, role_mode="logit")
