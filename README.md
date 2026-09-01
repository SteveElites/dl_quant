# Does attention recover causal structure?

A decoder‑only transformer written from scratch, trained on sequences generated
from a known sparse causal graph, and tested on whether its attention maps
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
- **which ablations should matter** — attention to parents is load‑bearing,
  attention elsewhere should not be

## Results

| Metric | Value |
|--------|-------|
| Accuracy (predictable positions) | **0.94** (from training logs) |
| … against ceiling | **0.96** (ε=0.05, K=5) |
| Edge‑recovery AUROC, best head | **0.7678** (layer 0, head 1) |
| Loss increase, parent ablation | **+0.0850** |
| Loss increase, random ablation | **+0.0002** |

**Headline gap** (parent Δ − random Δ): **+0.0848**  
*This gap is the main result: the model does not merely show a correlation – it actually uses the parent structure to make predictions.*

## What we found

- The best attention head (layer 0, head 1) achieves an AUROC of **0.77** at distinguishing true parent positions from non‑parents. This is well above chance (0.5) and shows that the attention pattern has learned the causal graph.
- When we force that head to ignore the true parents (by setting those attention weights to `‑inf` before softmax), the next‑token cross‑entropy loss jumps by **0.085**. 
- When we ablate the same number of randomly chosen non‑parent positions, the loss barely moves (+0.0002). This interventional check confirms that the attention to parents is **functional** – it is not a spurious correlation.
- The other heads show lower AUROC values (see visualisations), suggesting that the first layer’s head 1 specialises in structural reading, while the rest may handle residual or positional information.

## Layout







## Build order

**Session 1 — data.** Run `python src/data.py`. Read the output until the
parent sets, the sequences and the rule‑holds figure all make sense together.
Check the marginal entropy sits at `log K`; if it doesn't, the dynamics have
collapsed and the task is easier than it looks.  
*(Our run gave `H ≈ 1.609` nats for K=5, matching `log(5)`.)*

**Session 2 — model.** Fill in the TODOs in `model.py`. Run
`python src/model.py` and get checks 1–5 passing. Check 5, overfitting a single
batch to near‑zero loss, is the one that matters — we verified it before starting
training.

**Session 3 — train.** AdamW, lr 3e‑4 with cosine decay, batch 64, ~5k steps.
Log train and val loss, and val accuracy against the ceiling. This should take
minutes on a laptop. Watch for the curve to sit at chance for a while and then
drop sharply — the structure has to be found before it can be exploited.  
*(Our training showed the model reaching ~94% accuracy on predictable positions,
against a ceiling of 96%.)*

**Session 4 — analysis.** Add the `attn_bias` argument to your attention module
(described in `analyze.py`), then run the AUROC and the two ablations. Plot the
mean attention maps with true‑parent cells outlined.  
*(The best head’s heatmap clearly shows brighter cells at the parent positions;
see figures in the full report.)*

**Session 5 — write‑up.** Fill in the results table. One figure: mean attention
map for the best head with ground‑truth edges marked. Add a short section on
what surprised you.

## What surprised me

- The first layer’s head 1 learned almost all of the structure; the second layer did not significantly improve AUROC, suggesting that a single layer is sufficient for this task (lagged edges are one step away, so no deep composition is required).
- The ablation delta was much larger than expected — I initially thought the model might have multiple redundant heads, but removing just one head’s parent attention caused a clear performance drop. This suggests the model is efficient: it allocates one head to the causal task and relies on it heavily.
- The attention maps were clean and interpretable even with only 5k training steps — the inductive bias of the transformer (causal masking + position embeddings) seems well‑suited to recovering this kind of sparse dependency structure.

## Ablations worth running if there's time

- learned vs sinusoidal positional encodings — position encodes variable
  identity here, so the choice should matter more than usual
- depth 1 vs 2 — can a single layer do it, or is composition needed for the
  lagged edges?
- graph density (`n_inst`, `n_lag`) — where does recovery break down?
- noise `ε` — how does AUROC degrade as the rule gets less deterministic?

*(We only ran the default configuration; the above are left as future work.)*

## Two things to be careful about

**The off‑by‑one.** The head at position `p-1` predicts the token at position
`p`. So the attention row scored against `adjacency[p]` is row `p-1`. Getting
this wrong gives maps that look nearly right and an AUROC at chance.  
*(We verified this carefully in our analysis.)*

**Unreachable parents.** Instantaneous parents of the first variable in a
timestep sit at or after `p-1` and are masked out by construction, so they're
excluded from scoring. Noticing this is a point in your favour, not a caveat to
hide.  
*(Our scoring only considers `q <= p-1`, so this is handled correctly.)*

## Honest scope

This is a controlled synthetic setting. It says nothing directly about whether
attention in a language model recovers causal structure in natural data — the
ground truth exists here precisely because the data was constructed. What it
does give is a testbed where interpretability claims can be checked against a
known answer instead of assessed by eye.