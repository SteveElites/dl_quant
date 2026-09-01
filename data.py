"""
Synthetic sequences generated from a known sparse causal graph.

Layout
------
d variables, alphabet {0..K-1}, T timesteps.
Token at flat position p = t*d + i  holds the value of variable i at time t.

Generating rule, for t >= 1:
    x[t, i] = ( b_i + sum_j a_ij * x[t, j]  (instantaneous)
                    + sum_j c_ij * x[t-1, j] (lagged)     ) mod K   w.p. 1 - eps
            = uniform(K)                                            w.p. eps

K is prime and the coefficients a, c are nonzero, which keeps the marginal
distribution near-uniform. With K = 4 and unit coefficients the dynamics
collapse into a low-entropy attractor and a model can score well without
learning anything -- check entropy whenever you change K.

Instantaneous parents of variable i are restricted to j < i, which keeps the
within-timestep graph acyclic under the natural variable ordering.
Timestep 0 is drawn uniformly and is unpredictable by construction.

Everything here is numpy only. No torch needed.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CausalGraph:
    d: int
    K: int
    inst_parents: list          # inst_parents[i] -> list of j < i
    lag_parents: list           # lag_parents[i]  -> list of j in 0..d-1
    inst_coef: list             # matching nonzero coefficients mod K
    lag_coef: list
    bias: list

    def parent_positions(self, t: int, i: int):
        """Flat positions of the parents of variable i at time t.

        Returns [] for t == 0, where the value is unpredictable.
        """
        if t == 0:
            return []
        pos = [t * self.d + j for j in self.inst_parents[i]]
        pos += [(t - 1) * self.d + j for j in self.lag_parents[i]]
        return sorted(pos)

    def adjacency(self, T: int) -> np.ndarray:
        """Boolean [T*d, T*d] matrix. A[p, q] = True iff position q is a parent
        of position p. This is the ground truth your attention maps get scored
        against."""
        n = T * self.d
        A = np.zeros((n, n), dtype=bool)
        for t in range(T):
            for i in range(self.d):
                p = t * self.d + i
                for q in self.parent_positions(t, i):
                    A[p, q] = True
        return A


def sample_graph(d: int, K: int, n_inst: int = 1, n_lag: int = 1,
                 rng: np.random.Generator = None) -> CausalGraph:
    """Sample a random sparse graph.

    n_inst: max instantaneous parents per variable (capped by i, since j < i)
    n_lag:  exact number of lagged parents per variable
    """
    rng = rng or np.random.default_rng(0)
    inst_parents, lag_parents = [], []
    inst_coef, lag_coef, bias = [], [], []
    for i in range(d):
        k = min(n_inst, i)
        inst = sorted(rng.choice(i, size=k, replace=False).tolist()) if k > 0 else []
        lag = sorted(rng.choice(d, size=n_lag, replace=False).tolist())
        inst_parents.append(inst)
        lag_parents.append(lag)
        inst_coef.append(rng.integers(1, K, size=len(inst)).tolist())
        lag_coef.append(rng.integers(1, K, size=len(lag)).tolist())
        bias.append(int(rng.integers(0, K)))
    return CausalGraph(d=d, K=K, inst_parents=inst_parents, lag_parents=lag_parents,
                       inst_coef=inst_coef, lag_coef=lag_coef, bias=bias)


def generate(graph: CausalGraph, n_seq: int, T: int, eps: float = 0.05,
             rng: np.random.Generator = None) -> np.ndarray:
    """Return int64 array [n_seq, T*d] of token ids."""
    rng = rng or np.random.default_rng(0)
    d, K = graph.d, graph.K
    x = np.zeros((n_seq, T, d), dtype=np.int64)
    x[:, 0, :] = rng.integers(0, K, size=(n_seq, d))

    for t in range(1, T):
        for i in range(d):
            acc = np.full(n_seq, graph.bias[i], dtype=np.int64)
            for j, a in zip(graph.inst_parents[i], graph.inst_coef[i]):
                acc += a * x[:, t, j]
            for j, c in zip(graph.lag_parents[i], graph.lag_coef[i]):
                acc += c * x[:, t - 1, j]
            clean = acc % K
            noisy = rng.integers(0, K, size=n_seq)
            flip = rng.random(n_seq) < eps
            x[:, t, i] = np.where(flip, noisy, clean)

    return x.reshape(n_seq, T * d)


def bayes_accuracy(graph: CausalGraph, T: int, eps: float) -> dict:
    """Ceiling accuracy for next-token prediction.

    A perfect model still cannot beat this. Report your model against it --
    'we reach 0.94 against a ceiling of 0.96' is a real result; a bare
    accuracy number is not.

    Note the off-by-one: predicting the token at position p happens from
    position p-1, so the first target is p=1 and the final position is never
    a target.
    """
    d, K = graph.d, graph.K
    per_pos = []
    for p in range(1, T * d):
        t, i = divmod(p, d)
        if t == 0:
            per_pos.append(1.0 / K)            # unpredictable
        else:
            per_pos.append((1 - eps) + eps / K)
    per_pos = np.array(per_pos)
    predictable = np.array([divmod(p, d)[0] > 0 for p in range(1, T * d)])
    return {
        "overall": float(per_pos.mean()),
        "predictable_only": float(per_pos[predictable].mean()),
        "n_targets": int(len(per_pos)),
        "n_predictable": int(predictable.sum()),
    }


def make_dataset(d=5, K=5, T=12, n_train=20000, n_val=2000, eps=0.05,
                 n_inst=1, n_lag=1, seed=0):
    """One call to get everything the training script needs."""
    rng = np.random.default_rng(seed)
    graph = sample_graph(d, K, n_inst=n_inst, n_lag=n_lag, rng=rng)
    train = generate(graph, n_train, T, eps=eps, rng=rng)
    val = generate(graph, n_val, T, eps=eps, rng=rng)
    return {
        "graph": graph,
        "train": train,
        "val": val,
        "adjacency": graph.adjacency(T),
        "ceiling": bayes_accuracy(graph, T, eps),
        "config": dict(d=d, K=K, T=T, eps=eps, seq_len=T * d, vocab_size=K),
    }


if __name__ == "__main__":
    data = make_dataset()
    g, cfg = data["graph"], data["config"]
    print("config:", cfg)
    print("ceiling:", data["ceiling"])
    print("\nparent sets:")
    for i in range(g.d):
        print(f"  var {i}: inst={g.inst_parents[i]} coef={g.inst_coef[i]}  "
              f"lag={g.lag_parents[i]} coef={g.lag_coef[i]}  bias={g.bias[i]}")
    print("\nfirst sequence, reshaped [T, d]:")
    print(data["train"][0].reshape(cfg["T"], cfg["d"]))

    # sanity: does the rule actually hold at the stated rate?
    x = data["train"].reshape(-1, cfg["T"], cfg["d"])
    hits = tot = 0
    for t in range(1, cfg["T"]):
        for i in range(g.d):
            acc = g.bias[i]
            for j, a in zip(g.inst_parents[i], g.inst_coef[i]):
                acc = acc + a * x[:, t, j]
            for j, c in zip(g.lag_parents[i], g.lag_coef[i]):
                acc = acc + c * x[:, t - 1, j]
            hits += (x[:, t, i] == (acc % cfg["K"])).sum()
            tot += x.shape[0]
    print(f"\nrule holds on {hits/tot:.4f} of predictable tokens "
          f"(expect ~{(1 - cfg['eps']) + cfg['eps']/cfg['K']:.4f})")
    print("adjacency density:", data["adjacency"].mean().round(4))

    # sanity: marginal entropy should sit near log(K). If it does not, the
    # dynamics have collapsed and the task is easier than you think.
    counts = np.bincount(data["train"].ravel(), minlength=cfg["K"])
    q = counts / counts.sum()
    H = -(q * np.log(q + 1e-12)).sum()
    print(f"marginal entropy {H:.3f} nats (uniform = {np.log(cfg['K']):.3f}); "
          f"token freqs {q.round(3)}")
