"""
Does attention recover the causal graph, and does it matter?

Three results, in order of how much they're worth in an interview:

  1. accuracy vs the Bayes ceiling        -- did it learn the task at all
  2. edge-recovery AUROC from attention   -- observational: the pattern is there
  3. ablation delta                       -- interventional: the pattern is used

(3) is the one that separates this from a heatmap-squinting project. Correlation
between attention weight and true adjacency is cheap; showing that removing
attention to parent positions destroys the prediction, while removing an equal
number of random positions does not, is a claim about mechanism.
"""

import numpy as np
import torch


@torch.no_grad()
def accuracy(model, data, device="cpu", batch=256):
    """Next-token accuracy overall and on predictable positions only."""
    model.eval()
    d, T = data["config"]["d"], data["config"]["T"]
    val = torch.as_tensor(data["val"], device=device)
    correct, total = [], []
    for i in range(0, len(val), batch):
        idx = val[i:i + batch]
        logits, _ = model(idx)
        pred = logits[:, :-1].argmax(-1)
        correct.append((pred == idx[:, 1:]).cpu().numpy())
    c = np.concatenate(correct)                      # [N, L-1]
    # target at column p-1 is position p, so column j corresponds to position j+1
    predictable = np.array([(j + 1) // d > 0 for j in range(c.shape[1])])
    return {
        "overall": float(c.mean()),
        "predictable_only": float(c[:, predictable].mean()),
        "ceiling": data["ceiling"],
    }


@torch.no_grad()
def mean_attention(model, data, device="cpu", n=512):
    """Average attention maps over the validation set.

    Returns array [n_layers, n_heads, L, L]. Averaging is the right move here:
    the graph is fixed across sequences, so per-sequence noise should cancel and
    the structural pattern should survive.
    """
    model.eval()
    val = torch.as_tensor(data["val"][:n], device=device)
    acc = None
    for i in range(0, len(val), 128):
        _, attns = model(val[i:i + 128])
        stacked = torch.stack([a.mean(0) for a in attns])   # [Lay, H, L, L]
        acc = stacked if acc is None else acc + stacked
    return (acc / max(1, (len(val) + 127) // 128)).cpu().numpy()


def edge_auroc(attn_maps, adjacency, d):
    """AUROC of attention weight as a predictor of true parenthood.

    THE OFF-BY-ONE. adjacency[p, q] says q is a parent of position p. But the
    prediction of position p is made from row p-1. So the attention entry that
    should be large is attn[p-1, q], and you score attn[p-1, :] against
    adjacency[p, :].

    Scored only over q <= p-1, since the causal mask makes everything else
    structurally zero and including it would inflate the AUROC for free.
    Instantaneous parents of the first variable in a timestep are excluded for
    the same reason -- they sit at or after p-1 and are unreachable. Say so in
    the README; noticing it is a point in your favour.
    """
    L = adjacency.shape[0]
    scores, labels = [], []
    for p in range(1, L):
        row = p - 1
        for q in range(row + 1):
            scores.append(attn_maps[..., row, q])
            labels.append(adjacency[p, q])
    scores = np.stack(scores, axis=-1)          # [..., n_pairs]
    labels = np.array(labels, dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return None

    def _auroc(s):
        order = np.argsort(-s)
        y = labels[order]
        tp = np.cumsum(y)
        fp = np.cumsum(~y)
        return float(np.trapz(tp / tp[-1], fp / fp[-1]))

    flat = scores.reshape(-1, scores.shape[-1])
    per_head = np.array([_auroc(s) for s in flat]).reshape(scores.shape[:-1])
    return {
        "per_head": per_head,
        "best_head": float(per_head.max()),
        "best_head_index": np.unravel_index(per_head.argmax(), per_head.shape),
        "mean_head": float(per_head.mean()),
        "max_over_heads": _auroc(scores.max(axis=tuple(range(scores.ndim - 1)))),
    }


def ablation(model, data, adjacency, device="cpu", n=512, seed=0):
    """Interventional check.

    For the best head: zero its attention to true-parent positions, renormalise,
    measure the loss increase. Then do the same for an equal number of randomly
    chosen allowed positions per row. Report both.

    A large parent-delta and a small random-delta is the result you want. If
    they're comparable, the head is not doing what the AUROC suggested, and
    saying so honestly is a better README than a fudged win.

    TODO: this needs a forward hook that masks attention inside the block before
    the values are mixed. Two options:
      (a) add an optional `attn_bias` argument to MultiHeadSelfAttention.forward
          that is added to the scores pre-softmax (-inf at ablated positions);
      (b) register a hook that rewrites the post-softmax weights and renormalises.
    (a) is cleaner and is what real interp code does. Implement it in model.py
    and it costs you three lines there and a loop here.
    """
    raise NotImplementedError("wire up attn_bias in model.py first")


if __name__ == "__main__":
    from data import make_dataset
    data = make_dataset()
    print("ceiling:", data["ceiling"])
    print("adjacency:", data["adjacency"].shape,
          "density", round(float(data["adjacency"].mean()), 4))
