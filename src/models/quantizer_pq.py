"""
M2: task-agnostic Product-Quantization tokenizer (offline K-means + learnable embeddings).

Pipeline for one frame x in R^1024:
    x --[optional per-dim standardize]--> split into G groups of d=1024/G
       --[per-group K-means: nearest centroid]--> G integer codes
       --[per-group learnable embedding table: id -> vector]--> G vectors
       --[average fuse]--> one fused vector (dim = embed_dim) per frame

Key design (LOCKED, must be identical in M3 except how the codebook is learned):
  * G groups, V codes/group, average fusion, per-group learnable embedding tables.
  * The ONLY thing that differs in M3: the codebook is trained end-to-end by the
    translation loss (Gumbel-Softmax) instead of fit offline by K-means here.
  * Standardization (per-dim) is part of the shared front-end: if used, M1/M2/M3
    all use the same fitted scaler.

The K-means codebooks are FROZEN after offline fitting (task-agnostic). Only the
embedding tables are trainable, so the decoder can learn what each code "means".
Sequence length is preserved (per-frame op); the mask passes through unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ProductQuantizerM2(nn.Module):
    def __init__(self, G=2, V=320, feat_dim=1024, embed_dim=1024, standardize=True):
        super().__init__()
        if feat_dim % G != 0:
            raise ValueError(f"feat_dim={feat_dim} not divisible by G={G}")
        self.G, self.V = G, V
        self.feat_dim, self.embed_dim = feat_dim, embed_dim
        self.sub_dim = feat_dim // G
        self.standardize = standardize

        # Per-group K-means centroids (frozen buffers, filled by fit()).
        # shape (G, V, sub_dim)
        self.register_buffer("centroids", torch.zeros(G, V, self.sub_dim))
        # Per-dim standardization stats (frozen buffers, filled by fit()).
        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_std", torch.ones(feat_dim))

        # Learnable per-group embedding tables: code id -> embed_dim vector.
        self.embeddings = nn.ModuleList(
            [nn.Embedding(V, embed_dim) for _ in range(G)]
        )
        self._fitted = False

    # ---------- offline fitting (task-agnostic) ----------
    @torch.no_grad()
    def fit(self, feats_np, n_init=4, seed=0, verbose=True):
        """Fit per-dim scaler (optional) and per-group K-means on cached frames.

        feats_np: (N_frames, feat_dim) numpy array of WavLM features (train only).
        """
        X = np.asarray(feats_np, dtype=np.float64)
        if self.standardize:
            scaler = StandardScaler().fit(X)
            self.feat_mean.copy_(torch.tensor(scaler.mean_, dtype=torch.float32))
            self.feat_std.copy_(torch.tensor(scaler.scale_, dtype=torch.float32))
            X = scaler.transform(X)

        d = self.sub_dim
        for g in range(self.G):
            sub = X[:, g * d:(g + 1) * d]
            km = KMeans(n_clusters=self.V, n_init=n_init, random_state=seed).fit(sub)
            self.centroids[g].copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
            if verbose:
                counts = np.bincount(km.labels_, minlength=self.V)
                print(f"  group {g}: fitted {self.V} centroids | active={int((counts>0).sum())}/{self.V}")
        self._fitted = True
        return self

    # ---------- encoding: features -> code ids ----------
    @torch.no_grad()
    def encode(self, feats):
        """(B, T, feat_dim) -> (B, T, G) integer code ids. No grad (assignment is argmin)."""
        if not self._fitted:
            raise RuntimeError("call fit() before using the quantizer")
        B, T, D = feats.shape
        x = (feats - self.feat_mean) / self.feat_std if self.standardize else feats
        d = self.sub_dim
        codes = torch.empty(B, T, self.G, dtype=torch.long, device=feats.device)
        for g in range(self.G):
            sub = x[..., g * d:(g + 1) * d]                 # (B, T, d)
            cen = self.centroids[g].to(feats.device)        # (V, d)
            # squared euclidean distance to each centroid -> nearest
            # (B,T,1,d) - (V,d) -> (B,T,V,d) ; sum over d
            dist = ((sub.unsqueeze(2) - cen) ** 2).sum(-1)   # (B, T, V)
            codes[..., g] = dist.argmin(-1)
        return codes

    # ---------- forward: features -> fused embedding (trainable) ----------
    def forward(self, feats, mask=None):
        """(B, T, feat_dim) -> (fused (B, T, embed_dim), codes (B,T,G), mask).

        Codes come from frozen K-means (no grad); embeddings are learnable, so
        gradient flows into the embedding tables (and, in M3, into the codebook).
        """
        codes = self.encode(feats)                          # (B, T, G) long
        fused = 0.0
        for g in range(self.G):
            emb_g = self.embeddings[g](codes[..., g])        # (B, T, embed_dim)
            fused = fused + emb_g
        fused = fused / self.G                                # average fusion
        if mask is not None:
            fused = fused * mask.unsqueeze(-1).to(fused.dtype)
        return fused, codes, mask

    @torch.no_grad()
    def codebook_perplexity(self, feats):
        """Diagnostic: average codebook perplexity over a batch (utilization)."""
        codes = self.encode(feats)                           # (B,T,G)
        ppls = []
        for g in range(self.G):
            c = codes[..., g].flatten()
            counts = torch.bincount(c, minlength=self.V).float()
            p = counts / counts.sum()
            nz = p[p > 0]
            ppls.append(float(torch.exp(-(nz * nz.log()).sum())))
        return sum(ppls) / len(ppls)
