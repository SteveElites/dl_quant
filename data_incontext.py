r"""
In-context version of the task: a FRESH causal graph for every sequence.

Why this exists
---------------
In data.py the graph is fixed across the whole dataset, so every parent sits at
the same offset in every sequence. That makes "recover the causal graph" and
"memorise two fixed offsets" the same problem -- and an offset-only baseline
scores AUROC ~0.97 on it, well above what the trained model achieved. The fixed
setting cannot separate the two hypotheses, so no result from it is evidence of
structure learning.

Here each sequence gets its own graph, written into a token prefix that the
model must read. Parent offsets now vary sequence to sequence, so any
position-only strategy is at chance by construction. To score above chance the
model has to route attention according to the prefix -- which is the claim the
project is actually about.

Sequence layout
---------------
    [ inst_pa(0), lag_pa(0), inst_pa(1), lag_pa(1), ... , SEP, x_0 ... x_T ]
      \________________ 2d prefix tokens _______________/  ^
                                                     n_prefix = 2d + 1

Variable i's parent info always sits at prefix positions 2i and 2i+1, so the
prefix is positionally addressable; what varies is its CONTENT.

Vocabulary
----------
    0 .. K-1        variable values
    K .. K+d-1      variable-index tokens, IDX(j) = K + j
    K+d             SEP
    K+d+1           NONE  (variable 0 has no instantaneous parent)
    vocab_size = K + d + 2

Generating rule for t >= 1, unit coefficients by default so that the thing to
be inferred is parent IDENTITY rather than identity plus coefficients:

    x[t, i] = ( sum inst parents at t + sum lag parents at t-1 ) mod K   w.p. 1-eps
            = uniform(K)                                                 w.p. eps
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Vocab:
    K: int
    d: int

    @property
    def IDX0(self): return self.K
    @property
    def SEP(self): return self.K + self.d
    @property
    def NONE(self): return self.K + self.d + 1
    @property
    def size(self): return self.K + self.d + 2

    def idx(self, j): return self.K + j


def n_prefix(d: int) -> int:
    return 2 * d + 1


def sample_graphs(n_seq: int, d: int, rng: np.random.Generator):
    """One graph per sequence.

    inst_pa[s, i] = j < i, or -1 for none.   lag_pa[s, i] = j in 0..d-1.
    Instantaneous parents restricted to j < i keeps the within-step graph acyclic.
    """
    inst_pa = np.full((n_seq, d), -1, dtype=np.int64)
    lag_pa = np.zeros((n_seq, d), dtype=np.int64)
    for i in range(d):
        if i > 0:
            inst_pa[:, i] = rng.integers(0, i, size=n_seq)
        lag_pa[:, i] = rng.integers(0, d, size=n_seq)
    return inst_pa, lag_pa


def generate(n_seq: int, d: int = 5, K: int = 5, T: int = 16, eps: float = 0.05,
             rng: np.random.Generator = None):
    """Return (tokens [n_seq, L], inst_pa, lag_pa)."""
    rng = rng or np.random.default_rng(0)
    V = Vocab(K, d)
    inst_pa, lag_pa = sample_graphs(n_seq, d, rng)

    x = np.zeros((n_seq, T, d), dtype=np.int64)
    x[:, 0, :] = rng.integers(0, K, size=(n_seq, d))
    rows = np.arange(n_seq)
    for t in range(1, T):
        for i in range(d):
            acc = x[rows, t - 1, lag_pa[:, i]].copy()
            has_inst = inst_pa[:, i] >= 0
            if has_inst.any():
                j = np.where(has_inst, inst_pa[:, i], 0)
                acc = acc + np.where(has_inst, x[rows, t, j], 0)
            clean = acc % K
            flip = rng.random(n_seq) < eps
            x[:, t, i] = np.where(flip, rng.integers(0, K, size=n_seq), clean)

    pre = np.zeros((n_seq, 2 * d + 1), dtype=np.int64)
    for i in range(d):
        pre[:, 2 * i] = np.where(inst_pa[:, i] >= 0,
                                 V.IDX0 + np.maximum(inst_pa[:, i], 0), V.NONE)
        pre[:, 2 * i + 1] = V.IDX0 + lag_pa[:, i]
    pre[:, -1] = V.SEP

    return np.concatenate([pre, x.reshape(n_seq, T * d)], axis=1), inst_pa, lag_pa


def adjacency_for(inst_pa_s, lag_pa_s, d: int, T: int) -> np.ndarray:
    """Ground-truth [L, L] parent matrix for ONE sequence.

    A[p, q] = True iff data position q is a parent of data position p.
    Positions are absolute, i.e. they include the prefix offset.
    """
    npre = n_prefix(d)
    L = npre + T * d
    A = np.zeros((L, L), dtype=bool)
    for t in range(1, T):
        for i in range(d):
            p = npre + t * d + i
            A[p, npre + (t - 1) * d + lag_pa_s[i]] = True
            if inst_pa_s[i] >= 0:
                A[p, npre + t * d + inst_pa_s[i]] = True
    return A


def predictable_targets(d: int, T: int) -> np.ndarray:
    """Boolean over target columns (target at column c is position c+1).

    True only for data positions with t >= 1. Prefix tokens and the t=0 block
    are unpredictable by construction and are excluded from both the reported
    accuracy and the ceiling -- averaging them in just drags every number toward
    chance and hides what the model is doing.
    """
    npre = n_prefix(d)
    L = npre + T * d
    keep = np.zeros(L - 1, dtype=bool)
    for c in range(L - 1):
        p = c + 1
        if p >= npre and (p - npre) // d >= 1:
            keep[c] = True
    return keep


def bayes_accuracy(K: int, eps: float) -> float:
    return (1 - eps) + eps / K


def bayes_loss(K: int, eps: float) -> float:
    p = 1 - eps + eps / K
    q = eps / K
    return float(-(p * np.log(p) + (K - 1) * q * np.log(q)))


def make_dataset(d=5, K=5, T=16, n_train=40000, n_val=4000, eps=0.05, seed=0):
    rng = np.random.default_rng(seed)
    tr, tr_i, tr_l = generate(n_train, d, K, T, eps, rng)
    va, va_i, va_l = generate(n_val, d, K, T, eps, rng)
    V = Vocab(K, d)
    return {
        "train": tr, "val": va,
        "val_inst_pa": va_i, "val_lag_pa": va_l,
        "train_inst_pa": tr_i, "train_lag_pa": tr_l,
        "predictable": predictable_targets(d, T),
        "config": dict(d=d, K=K, T=T, eps=eps, n_prefix=n_prefix(d),
                       seq_len=n_prefix(d) + T * d, vocab_size=V.size),
        "ceiling": {"accuracy": bayes_accuracy(K, eps), "loss": bayes_loss(K, eps)},
    }


if __name__ == "__main__":
    data = make_dataset(n_train=4000, n_val=1000)
    cfg = data["config"]
    print("config :", cfg)
    print("ceiling:", {k: round(v, 4) for k, v in data["ceiling"].items()},
          "(predictable data positions only)")

    V = Vocab(cfg["K"], cfg["d"])
    s = data["train"][0]
    print(f"\nsequence 0 prefix: {s[:cfg['n_prefix']].tolist()}  "
          f"(SEP={V.SEP}, NONE={V.NONE}, IDX(j)={V.IDX0}+j)")
    print("parents:", [(int(data['train_inst_pa'][0, i]),
                        int(data['train_lag_pa'][0, i])) for i in range(cfg['d'])])

    # graphs must actually differ across sequences, or nothing has changed
    uniq = len({(tuple(data["train_inst_pa"][s]), tuple(data["train_lag_pa"][s]))
                for s in range(len(data["train"]))})
    print(f"\ndistinct graphs in 4000 sequences: {uniq}")

    # entropy of the data region
    npre = cfg["n_prefix"]
    vals = data["train"][:, npre:].ravel()
    q = np.bincount(vals, minlength=cfg["K"])[:cfg["K"]] / len(vals)
    H = -(q * np.log(q + 1e-12)).sum()
    print(f"marginal entropy {H:.3f} nats (uniform = {np.log(cfg['K']):.3f})")

    # rule check
    x = data["train"][:, npre:].reshape(-1, cfg["T"], cfg["d"])
    ip, lp = data["train_inst_pa"], data["train_lag_pa"]
    r = np.arange(len(x)); hit = tot = 0
    for t in range(1, cfg["T"]):
        for i in range(cfg["d"]):
            acc = x[r, t - 1, lp[:, i]].copy()
            m = ip[:, i] >= 0
            acc = acc + np.where(m, x[r, t, np.maximum(ip[:, i], 0)], 0)
            hit += (x[:, t, i] == acc % cfg["K"]).sum(); tot += len(x)
    print(f"rule holds on {hit/tot:.4f} (expect ~{bayes_accuracy(cfg['K'], cfg['eps']):.4f})")


def lag_candidate_pairs(d: int, T: int):
    """The scoring set that actually tests structure learning.

    Yields (target_p, attention_row, candidate_q) restricted to the d cells of
    the previous timestep block -- the exact set the lag parent is drawn from.

    This restriction is not a detail, it is the metric. Scoring over ALL allowed
    cells lets a model win by attending to recent positions: parents are always
    within the last ~d tokens while most negatives are far away, so a pure
    recency heuristic reaches AUROC ~0.95 without reading anything. Restricted
    to the candidate block, offset-only and recency-only baselines both sit at
    chance (~0.53 and ~0.49), so anything above 0.5 has to come from the prefix.
    """
    npre = n_prefix(d)
    for t in range(1, T):
        for i in range(d):
            p = npre + t * d + i
            row = p - 1
            for j in range(d):
                q = npre + (t - 1) * d + j
                if q <= row:
                    yield p, row, q
