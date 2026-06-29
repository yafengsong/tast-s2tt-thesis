"""
Intrinsic PQ quality sweep on cached WavLM-L21 features.

Answers, WITHOUT any downstream training (valid on the 200-sample dev set since
these are properties of the feature distribution):
  1. How does product-quantization reconstruction error change with G?
  2. Does PCA rotation before splitting help at each G? (RPQ paper reports
     WavLM-L21 dims are weakly correlated -> rotation may be unnecessary.)
  3. (optional) Is the flat MSE-vs-G trend a scale artifact of a few
     high-variance dimensions? -> the `standardize` flag tests this.

Diagnostics per config:
  - mse  : raw reconstruction error (scale depends on feature variance).
  - nmse : mse / variance-of-quantized-features. Interpretable: ~fraction of
           variance LOST. nmse=0 perfect, nmse=1 useless. (1 - nmse) = variance kept.
  - active-code count + perplexity : codebook utilization (spots dead/wasted codes).

NOTE: G must divide 1024 evenly. Valid: 2,4,8,16,32 (-> 512,256,128,64,32 dims).
6 is INVALID (1024/6 is not integer).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pq_quality(X, G, V=320, rotate=False, standardize=False, seed=0, n_init=4):
    """Fit product-quantization (K-means per group) and report quality + utilization.

    Transforms are applied in order: standardize -> rotate. The nmse reference
    variance is taken from the *transformed* features (what is actually quantized),
    so nmse stays comparable across conditions. (PCA rotation preserves total
    variance; standardization changes it, hence the reference must follow Xin.)
    """
    D = X.shape[1]
    if D % G != 0:
        raise ValueError(f"D={D} not divisible by G={G}")
    d = D // G

    Xin = X
    if standardize:
        Xin = StandardScaler().fit_transform(Xin)        # per-dim zero mean, unit var
    if rotate:
        Xin = PCA(n_components=D, svd_solver="full", random_state=seed).fit_transform(Xin)

    var_ref = float(Xin.var())   # reference matches the features actually quantized

    recon = np.zeros_like(Xin)
    active_per_group, ppl_per_group = [], []
    for g in range(G):
        sub = Xin[:, g * d:(g + 1) * d]
        km = KMeans(n_clusters=V, n_init=n_init, random_state=seed).fit(sub)
        recon[:, g * d:(g + 1) * d] = km.cluster_centers_[km.labels_]
        counts = np.bincount(km.labels_, minlength=V).astype(float)
        p = counts / counts.sum()
        active_per_group.append(int((counts > 0).sum()))
        nz = p[p > 0]
        ppl_per_group.append(float(np.exp(-(nz * np.log(nz)).sum())))

    mse = float(np.mean((Xin - recon) ** 2))
    return {
        "G": G, "V": V, "rotate": rotate, "standardize": standardize,
        "mse": round(mse, 6),
        "nmse": round(mse / var_ref, 4),              # fraction of variance lost
        "var_kept": round(1.0 - mse / var_ref, 4),    # fraction of variance kept
        "bits_per_frame": G * (V.bit_length() - 1),
        "active_codes_per_group": active_per_group,
        "avg_active": round(float(np.mean(active_per_group)), 1),
        "avg_perplexity": round(float(np.mean(ppl_per_group)), 1),
    }


def run_sweep(X, G_values=(2, 4, 8, 16), V=320, rotate_values=(False, True),
              standardize=False, seed=0):
    rows = []
    for G in G_values:
        for rotate in rotate_values:
            r = pq_quality(X, G, V=V, rotate=rotate, standardize=standardize, seed=seed)
            rows.append(r)
            print(f"G={G:2d} | rot={str(r['rotate']):5s} | std={str(standardize):5s} "
                  f"| MSE={r['mse']:.4f} | nmse={r['nmse']:.4f} "
                  f"(kept {100*r['var_kept']:.1f}%) "
                  f"| active={r['avg_active']:5.1f}/{V} | ppl={r['avg_perplexity']:6.1f} "
                  f"| bits={r['bits_per_frame']}")
    return rows


def plot_sweep(rows, out_path="pq_sweep.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Gs = sorted({r["G"] for r in rows})
    rotate_opts = sorted({r["rotate"] for r in rows})

    def series(rotate, key):
        return [next(r[key] for r in rows if r["G"] == G and r["rotate"] == rotate)
                for G in Gs]

    style = {False: ("o-", "no rotation"), True: ("s--", "PCA rotation")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: normalized MSE (fraction of variance lost) -- interpretable y-axis
    for rot in rotate_opts:
        fmt, lab = style[rot]
        ax1.plot(Gs, series(rot, "nmse"), fmt, label=lab)
    ax1.set_xlabel("G (number of groups)")
    ax1.set_ylabel("nMSE  (fraction of variance lost)")
    ax1.set_title("PQ reconstruction loss vs G")
    ax1.set_xticks(Gs); ax1.legend(); ax1.grid(alpha=0.3)

    # Right: codebook utilization
    V = rows[0]["V"]
    for rot in rotate_opts:
        fmt, lab = style[rot]
        ax2.plot(Gs, series(rot, "avg_active"), fmt, label=lab)
    ax2.axhline(V, color="grey", ls=":", label=f"V={V} (full use)")
    ax2.set_xlabel("G (number of groups)")
    ax2.set_ylabel("avg active codes / group")
    ax2.set_title("Codebook utilization vs G")
    ax2.set_xticks(Gs); ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print("saved plot ->", out_path)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((35000, 1024)).astype(np.float32)
    rows = run_sweep(X, G_values=(2, 4, 8, 16), V=320, seed=0)
    plot_sweep(rows)
