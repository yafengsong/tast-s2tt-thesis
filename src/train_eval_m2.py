"""
M2 raw-vs-standardized A/B helper (overfit proxy on the dev set).

build_and_eval_m2(standardize): fits a fresh PQ quantizer, trains
(embeddings + bridge LayerNorm + decoder LoRA) for a FIXED number of steps
(equal budget per arm), then reports final train loss, train-set sacreBLEU
(tok=zh), and codebook perplexity. Also records, WITHOUT stopping early, the
step at which loss first crossed `loss_threshold` as a secondary
convergence-speed signal. Everything except the `standardize` flag is held
identical so the comparison is fair.

Primary metric : final_loss + train_bleu at fixed steps -> "better with equal budget".
Secondary      : steps_to_threshold                      -> "converged faster".

NOTE: train==eval (overfit proxy). A near-tie means "inconclusive on 200,
defer to held-out data", NOT "standardization is irrelevant".
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def build_and_eval_m2(
    standardize,
    *,
    cached_feats, cached_masks, cached_targets,   # precomputed, frozen
    encoder, make_bridge, make_decoder,           # factories -> fresh modules
    ProductQuantizerM2, device,
    G=2, V=320, embed_dim=1024,
    steps=700, loss_threshold=0.15, batch=8,      # FIXED step budget
    lr=3e-4, seed=42, log_every=100,
):
    torch.manual_seed(seed); np.random.seed(seed)

    # 1. Fresh quantizer (the ONLY thing that varies between arms)
    X = np.concatenate([f.numpy() for f in cached_feats], axis=0)
    q = ProductQuantizerM2(G=G, V=V, feat_dim=X.shape[1], embed_dim=embed_dim,
                           standardize=standardize).to(device)
    q.fit(X, n_init=4, seed=seed, verbose=False)

    # 2. Fresh bridge + decoder so neither arm inherits the other's weights
    bridge = make_bridge().to(device)
    dec = make_decoder().to(device)

    # 3. Trainable params: quantizer embeddings + bridge LN + decoder LoRA
    trainable = (
        [p for p in q.embeddings.parameters() if p.requires_grad]
        + [p for p in bridge.parameters() if p.requires_grad]
        + [p for p in dec.parameters() if p.requires_grad]
    )
    opt = torch.optim.AdamW(trainable, lr=lr)

    N = len(cached_feats)
    def make_batch(idx):
        feats = pad_sequence([cached_feats[i].to(device) for i in idx], batch_first=True)
        masks = pad_sequence([cached_masks[i].to(device) for i in idx], batch_first=True)
        labels = dec.tokenize_targets([cached_targets[i] for i in idx], device=device)
        return feats, masks, labels

    # 4. FIXED-step training. Record threshold-crossing step but DO NOT stop.
    q.train(); bridge.train(); dec.model.train()
    perm = torch.randperm(N); ptr = 0
    steps_to_thresh = None
    last_loss = float("nan")
    for step in range(1, steps + 1):
        if ptr + batch > N:
            perm = torch.randperm(N); ptr = 0
        idx = perm[ptr:ptr + batch].tolist(); ptr += batch

        feats, masks, labels = make_batch(idx)
        fused, _codes, masks = q(feats, masks)
        memory, mmask = bridge(fused, masks)
        loss = dec(memory, mmask, labels).loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        last_loss = loss.item()
        if steps_to_thresh is None and last_loss <= loss_threshold:
            steps_to_thresh = step                       # record, keep training
        if step % log_every == 0 or step == 1:
            tag = f" (<thr at {steps_to_thresh})" if steps_to_thresh else ""
            print(f"  [std={standardize}] step {step:4d} | loss {last_loss:.4f}{tag}")

    # 5. Train-set BLEU (overfit proxy) on the same samples
    import sacrebleu
    q.eval(); bridge.eval(); dec.model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for i in range(0, N, batch):
            idx = list(range(i, min(i + batch, N)))
            feats, masks, _ = make_batch(idx)
            fused, _c, masks = q(feats, masks)
            memory, mmask = bridge(fused, masks)
            preds = dec.generate(memory, mmask, max_length=128, num_beams=1)
            hyps.extend(preds)
            refs.extend([cached_targets[j] for j in idx])
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score

    # codebook utilization on a representative batch
    probe = pad_sequence([cached_feats[i].to(device) for i in range(min(batch, N))],
                         batch_first=True)
    perp = float(q.codebook_perplexity(probe))

    result = {
        "standardize": standardize,
        "final_loss": round(last_loss, 4),       # PRIMARY
        "train_bleu": round(bleu, 2),            # PRIMARY
        "steps_to_threshold": steps_to_thresh,   # SECONDARY (None if never crossed)
        "codebook_perplexity": round(perp, 1),
        "total_steps": steps,
    }
    print(f"  [std={standardize}] RESULT: {result}\n")
    return result
