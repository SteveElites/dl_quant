"""
Does attention route according to the causal structure, and does it matter?

    python analyze.py --task incontext --ckpt model_incontext.pt

Reports, in order:

  1. accuracy vs the Bayes ceiling
  2. edge-recovery AUROC, ALONGSIDE two position-only baselines
  3. ablation: parent cells vs an attention-weight-MATCHED control

Two design points that the first version of this file got wrong, both of which
inflated the result:

  * Scoring over every allowed cell rewards recency. Parents always sit within
    the last ~d tokens while most negatives are far away, so "attend nearby"
    reaches AUROC ~0.95 having learned nothing. Scoring is therefore restricted
    to the candidate block the parent is actually drawn from, where position-only
    baselines sit at chance. The baselines are computed and printed every run --
    an AUROC that does not clear them is not a result.

  * Ablating randomly chosen non-parent cells removes almost no attention mass,
    so a near-zero effect there is close to a tautology. The control here matches
    the total attention mass removed, which is the comparison that can fail.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import CausalTransformer, NEG_INF

TRAPZ = getattr(np, "trapz", None) or np.trapezoid   # np.trapz removed in NumPy 2.0


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    return float(TRAPZ(tp / tp[-1], fp / fp[-1]))


@torch.no_grad()
def collect(model, val, d, T, n_seq, inst_pa, lag_pa, batch=64):
    """Per-sequence attention scored against that sequence's own graph.

    Attention maps are NOT averaged across sequences here: with a different
    graph per sequence, the mean map is meaningless.
    """
    from data_incontext import adjacency_for, lag_candidate_pairs
    pairs = list(lag_candidate_pairs(d, T))
    rows = np.array([r for _, r, _ in pairs])
    cols = np.array([q for _, _, q in pairs])

    per_head, labels_all, offsets = [], [], np.array([r - q for _, r, q in pairs])
    for i in range(0, n_seq, batch):
        b = val[i:i + batch]
        _, attns = model(b)
        A = torch.stack(attns, 0)                    # [Lay, B, H, L, L]
        sel = A[:, :, :, rows, cols].cpu().numpy()   # [Lay, B, H, P]
        per_head.append(sel)
        for s in range(b.size(0)):
            adj = adjacency_for(inst_pa[i + s], lag_pa[i + s], d, T)
            labels_all.append(adj[[p for p, _, _ in pairs], cols])
    sel = np.concatenate(per_head, axis=1)           # [Lay, N, H, P]
    labels = np.concatenate(labels_all)              # [N*P]
    n_lay, N, n_head, P = sel.shape
    flat = sel.transpose(0, 2, 1, 3).reshape(n_lay, n_head, N * P)
    return flat, labels, np.tile(offsets, N), (rows, cols, pairs)


def position_baselines(offsets, labels):
    rate = {o: labels[offsets == o].mean() for o in np.unique(offsets)}
    return {
        "offset_only": auroc([rate[o] for o in offsets], labels),
        "recency_only": auroc(-offsets, labels),
    }


@torch.no_grad()
def ablate(model, val, n_seq, vocab, layer, head, rows, cols, pairs,
           inst_pa, lag_pa, d, T, seed=0, batch=64):
    """Parent ablation vs an attention-mass-matched control."""
    from data_incontext import adjacency_for
    rng = np.random.default_rng(seed)
    n_heads = model.blocks[0].attn.n_heads
    L = model.seq_len
    base = par = ctl = 0.0
    n_tok = 0
    mass_par = mass_ctl = 0.0

    for i in range(0, n_seq, batch):
        b = val[i:i + batch]
        B = b.size(0)
        logits, attns = model(b)
        w = attns[layer][:, head].cpu().numpy()      # [B, L, L]

        bias_p = torch.zeros(B, n_heads, L, L, device=b.device)
        bias_c = torch.zeros(B, n_heads, L, L, device=b.device)
        for s in range(B):
            adj = adjacency_for(inst_pa[i + s], lag_pa[i + s], d, T)
            is_par = np.array([adj[p, q] for p, _, q in pairs])
            pr, pc = rows[is_par], cols[is_par]
            bias_p[s, head, pr, pc] = NEG_INF
            mass_par += w[s, pr, pc].sum()

            # match the removed mass using non-parent candidate cells
            nr, nc = rows[~is_par], cols[~is_par]
            wts = w[s, nr, nc]
            target = w[s, pr, pc].sum()
            order = rng.permutation(len(nr))
            got, chosen = 0.0, []
            for k in order:
                if got >= target:
                    break
                chosen.append(k)
                got += wts[k]
            chosen = np.array(chosen, dtype=int)
            if len(chosen):
                bias_c[s, head, nr[chosen], nc[chosen]] = NEG_INF
                mass_ctl += got

        tgt = b[:, 1:].reshape(-1)
        for bias, acc in ((None, "base"), (bias_p, "par"), (bias_c, "ctl")):
            lg, _ = model(b, attn_biases=[bias if j == layer else None
                                          for j in range(len(model.blocks))])
            v = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), tgt,
                                reduction="sum").item()
            if acc == "base": base += v
            elif acc == "par": par += v
            else: ctl += v
        n_tok += tgt.numel()

    return {"base": base / n_tok, "parent": par / n_tok, "matched": ctl / n_tok,
            "d_parent": (par - base) / n_tok, "d_matched": (ctl - base) / n_tok,
            "mass_parent": mass_par / n_seq, "mass_matched": mass_ctl / n_seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["incontext"], default="incontext")
    ap.add_argument("--ckpt", default="model_incontext.pt")
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    from data_incontext import make_dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = make_dataset(d=5, K=5, T=16, n_train=10, n_val=args.n_seq, seed=0)
    cfg = data["config"]
    d, T = cfg["d"], cfg["T"]

    ck = torch.load(args.ckpt, map_location=device)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    n_layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    model = CausalTransformer(cfg["vocab_size"], cfg["seq_len"], n_layers=n_layers).to(device)
    model.load_state_dict(sd)
    model.eval()

    val = torch.as_tensor(data["val"], dtype=torch.long, device=device)
    inst_pa, lag_pa = data["val_inst_pa"], data["val_lag_pa"]

    from train import evaluate
    vl, acc = evaluate(model, val, data["predictable"], cfg["vocab_size"])
    model.eval()
    print(f"val loss {vl:.4f} (ceiling {data['ceiling']['loss']:.4f}) | "
          f"acc {acc:.4f} (ceiling {data['ceiling']['accuracy']:.4f})")

    flat, labels, offsets, (rows, cols, pairs) = collect(
        model, val, d, T, args.n_seq, inst_pa, lag_pa)

    base = position_baselines(offsets, labels)
    print(f"\nposition-only baselines: offset {base['offset_only']:.4f}, "
          f"recency {base['recency_only']:.4f}  (chance 0.5)")

    n_lay, n_head, _ = flat.shape
    per_head = np.zeros((n_lay, n_head))
    for l in range(n_lay):
        for h in range(n_head):
            per_head[l, h] = auroc(flat[l, h], labels)
    print("AUROC per head:\n", np.round(per_head, 4))
    bl, bh = np.unravel_index(np.nanargmax(per_head), per_head.shape)
    best = per_head[bl, bh]
    print(f"best head: layer {bl}, head {bh} -> {best:.4f}")
    print("VERDICT:", "clears position-only baselines"
          if best > max(base.values()) + 0.02 else
          "DOES NOT clear position-only baselines -- no evidence of structure use")

    res = ablate(model, val, args.n_seq, cfg["vocab_size"], bl, bh,
                 rows, cols, pairs, inst_pa, lag_pa, d, T)
    print(f"\nbase loss        {res['base']:.4f}")
    print(f"parent ablation  {res['parent']:.4f}  (delta +{res['d_parent']:.4f}, "
          f"mass {res['mass_parent']:.2f}/seq)")
    print(f"matched control  {res['matched']:.4f}  (delta +{res['d_matched']:.4f}, "
          f"mass {res['mass_matched']:.2f}/seq)")
    print(f"headroom to no-structure: {np.log(cfg['K']) - data['ceiling']['loss']:.4f} nats")
    print(f"parent delta is {100*res['d_parent']/(np.log(cfg['K'])-data['ceiling']['loss']):.1f}% of headroom")

    if args.figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from data_incontext import adjacency_for
        with torch.no_grad():
            _, attns = model(val[:1])
        w = attns[bl][0, bh].cpu().numpy()
        adj = adjacency_for(inst_pa[0], lag_pa[0], d, T)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(w, cmap="magma")
        py, px = np.where(adj)
        ax.scatter(px, py - 1, s=8, facecolors="none", edgecolors="cyan", lw=0.6,
                   label="true parent (row p-1)")
        ax.set_title(f"layer {bl} head {bh} — AUROC {best:.3f} "
                     f"(baselines {max(base.values()):.3f})")
        ax.set_xlabel("key position"); ax.set_ylabel("query position")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout(); fig.savefig("figures/attention_best_head.png", dpi=150)
        print("wrote figures/attention_best_head.png")


if __name__ == "__main__":
    main()
