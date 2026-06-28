"""Bridge modules — the ONLY part that differs between M1/M2/M3.

M1 ContinuousBridge is implemented. M2 (offline PQ) and M3 (Gumbel PQ)
go here too, so the controlled comparison is enforced by code structure.
"""
import torch.nn as nn


class ContinuousBridge(nn.Module):
    """M1: project frozen WavLM features into NLLB space. No quantization."""
    def __init__(self, in_dim=1024, out_dim=1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feats):           # (B, T, in_dim)
        return self.norm(self.proj(feats))


# class OfflinePQBridge(nn.Module):   # M2 — to implement
# class GumbelPQBridge(nn.Module):    # M3 — to implement
