# Does attention recover causal structure?

A decoder-only transformer written from scratch, trained on sequences generated
from a known sparse causal graph, then tested on whether its attention maps
recover the generating structure — and whether that structure is actually used.

## The setup

`d` discrete variables over a prime alphabet `{0..K-1}`, unrolled for `T`
timesteps and flattened so that position `p = t*d + i` holds variable `i` at
time `t`. Each variable has a sparse parent set: some instantaneous (earlier
variables in the same timestep), some lagged (any variable at `t-1`). Values
follow a random linear rule mod `K`, corrupted with probability `ε`.

Because the graph is known, three things are known that usually aren't:

- **the Bayes ceiling** — `(1-ε) + ε/K` on predictable positions, so model
  accuracy can be reported against what is actually achievable
- **the target attention pattern** — to predict position `p`, the model must
  read the positions holding `p`'s parents
- **which ablations should matter** — attention to parents is load-bearing,
  attention elsewhere should not be

## Results

| | |
|---|---|
| Accuracy (predictable positions) | — / ceiling — |
| Edge-recovery AUROC, best head | — |
| Loss increase, parent ablation | — |
| Loss increase, random ablation | — |

*(fill these in; the gap between the last two rows is the headline)*

## Layout

```
src/data.py      generator, ground-truth adjacency, Bayes ceiling   
src/model.py     attention, block, transformer                      
src/train.py     training loop, logging                             
src/analyze.py   AUROC, ablation, figures                           
```

## Build order

**Session 1 — data.** Run `python src/data.py`. Read the output until the
parent sets, the sequences and the rule-holds figure all make sense together.
Check the marginal entropy sits at `log K`; if it doesn't, the dynamics have
collapsed and the task is easier than it looks.

**Session 2 — model.** Fill in the TODOs in `model.py`. Run
`python src/model.py` and get checks 1–5 passing. Check 5, overfitting a single
batch to near-zero loss, is the one that matters — do not start training on real
data until it passes.

**Session 3 — train.** AdamW, lr 3e-4 with cosine decay, batch 64, ~5k steps.
Log train and val loss, and val accuracy against the ceiling. This should take
minutes on a laptop. Watch for the curve to sit at chance for a while and then
drop sharply — the structure has to be found before it can be exploited.

**Session 4 — analysis.** Add the `attn_bias` argument to your attention module
(described in `analyze.py`), then run the AUROC and the two ablations. Plot the
mean attention maps with true-parent cells outlined.

**Session 5 — write-up.** Fill in the results table. One figure: mean attention
map for the best head with ground-truth edges marked. Add a short section on
what surprised you.

## Ablations worth running if there's time

- learned vs sinusoidal positional encodings — position encodes variable
  identity here, so the choice should matter more than usual
- depth 1 vs 2 — can a single layer do it, or is composition needed for the
  lagged edges?
- graph density (`n_inst`, `n_lag`) — where does recovery break down?
- noise `ε` — how does AUROC degrade as the rule gets less deterministic?

## Two things to be careful about

**The off-by-one.** The head at position `p-1` predicts the token at position
`p`. So the attention row scored against `adjacency[p]` is row `p-1`. Getting
this wrong gives maps that look nearly right and an AUROC at chance.

**Unreachable parents.** Instantaneous parents of the first variable in a
timestep sit at or after `p-1` and are masked out by construction, so they're
excluded from scoring. Noticing this is a point in your favour, not a caveat to
hide.

## Honest scope

This is a controlled synthetic setting. It says nothing directly about whether
attention in a language model recovers causal structure in natural data — the
ground truth exists here precisely because the data was constructed. What it
does give is a testbed where interpretability claims can be checked against a
known answer instead of assessed by eye.
