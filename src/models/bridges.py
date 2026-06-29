"""
Encoder->decoder bridge. Shared, identical across M1/M2/M3.

Maps frozen WavLM-Large features (1024-dim) into a representation the
NLLB-200 decoder cross-attention can consume.

Design decisions (locked for RQ2 fairness):
  * LayerNorm: always on. Cheap insurance against WavLM feature-scale
    mismatch with NLLB's expected encoder-memory statistics. No length
    dependence, ~2*1024 params.
  * No Linear(1024->1024): WavLM hidden size == NLLB d_model == 1024, so
    no projection is needed for dimension. Representation-space adaptation
    is handled by the decoder LoRA, not here. Keeping a projection here
    would add an untracked variable across M1/M2/M3.
  * Positional embedding: sinusoidal (no params, no max-length cap),
    toggleable via `use_pos_embed` so the with/without ablation is a pure
    config switch. Whatever is chosen MUST be identical across M1/M2/M3.

The bridge does NOT change sequence length, so the frame mask from the
encoder is passed through unchanged.
"""
import math

import torch
import torch.nn as nn


def sinusoidal_position_encoding(seq_len: int, dim: int,
                                 device=None, dtype=torch.float32) -> torch.Tensor:
    """Standard Transformer sinusoidal positional encoding.

    Returns (seq_len, dim). No parameters, no length ceiling.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for sinusoidal PE, got {dim}")
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )  # (dim/2,)
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (T, dim)


class EncoderDecoderBridge(nn.Module):
    """LayerNorm (+ optional sinusoidal pos-embed) bridge. Mask passthrough.

    Args:
        hidden_dim: feature dim of encoder output == NLLB d_model. Default 1024.
        use_pos_embed: if True, add sinusoidal positional encoding after the
            LayerNorm. This is the single ablation switch.
    """

    def __init__(self, hidden_dim: int = 1024, use_pos_embed: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_pos_embed = use_pos_embed
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            features: (B, T, hidden_dim) WavLM layer-21 features.
            mask:     (B, T) frame mask (1 = real, 0 = padding).
        Returns:
            (features_out, mask) with the SAME shapes. Mask is unchanged
            because the bridge does not alter sequence length.
        """
        if features.dim() != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(features.shape)}")
        if features.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"feature dim {features.shape[-1]} != hidden_dim {self.hidden_dim}"
            )

        x = self.layer_norm(features)  # (B, T, D)

        if self.use_pos_embed:
            T = x.shape[1]
            pe = sinusoidal_position_encoding(
                T, self.hidden_dim, device=x.device, dtype=x.dtype
            )  # (T, D)
            x = x + pe.unsqueeze(0)  # broadcast over batch

        # Zero out padding positions so downstream never sees PE-on-padding noise.
        # (Cross-attention also masks these, but this keeps the memory clean.)
        x = x * mask.unsqueeze(-1).to(x.dtype)

        return x, mask