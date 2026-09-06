import copy

import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class InContextTransformerLayer(nn.Module):
    """Transformer layer that processes all nodes in parallel with optional RoPE support."""

    def __init__(
        self,
        node_dim: int,
        num_heads: int = 8,
        ff_dim: int = None,
        attn_layer: nn.Module = None,
        use_rms_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * node_dim

        self.num_heads = num_heads

        self.attn_layer = attn_layer

        self.feed_forward = nn.Sequential(
            nn.Linear(node_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, node_dim),
            nn.Dropout(dropout),
        )

        # Choose normalization type
        if use_rms_norm:
            self.norm1 = LlamaRMSNorm(node_dim)
            self.norm2 = LlamaRMSNorm(node_dim)
        else:
            self.norm1 = nn.LayerNorm(node_dim)
            self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, num_nodes, seq_len, node_dim]
            mask: [seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len] causal mask
        Returns:
            output: [batch_size, num_nodes, seq_len, node_dim]
        """

        # Self-attention with residual connection
        attn_out = self.attn_layer(x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class InContextTransformer(nn.Module):
    """Transformer that processes temporal sequences for all nodes in parallel with optional RoPE support."""

    def __init__(
        self,
        node_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = None,
        use_rms_norm: bool = False,
        attn_layer: nn.Module = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                InContextTransformerLayer(node_dim, num_heads, ff_dim, copy.deepcopy(attn_layer), use_rms_norm, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_mask=None):
        """
        Args:
            x: [batch_size, num_nodes, seq_len, node_dim]
            src_mask: [seq_len, seq_len] or [batch_size, num_heads, seq_len, seq_len] causal mask
        Returns:
            output: [batch_size, num_nodes, seq_len, node_dim]
        """

        for layer in self.layers:
            x = layer(x, src_mask)
        return x
