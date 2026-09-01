# Does attention recover causal structure?

A decoder-only transformer written from scratch, trained on sequences generated
from a known causal graph, then tested on whether its attention routes according
to that graph — and whether the routing is load-bearing.

The headline finding is negative and the negative result is the point: the
obvious version of this experiment cannot distinguish structure learning from
positional heuristics, and most of the work here is building a measurement that
can.

## Setup

`d` discrete variables over a prime alphabet `{0..K-1}`, unrolled for `T`
timesteps and flattened so position `p = t*d + i` holds variable `i` at time `t`.
Each variable has one instantaneous parent (an earlier variable in the same
timestep) and one lagged parent (any variable at `t-1`). Values follow a linear
rule mod `K`, corrupted with probability `ε`.

Two variants:

| | `data.py` | `data_incontext.py` |
|---|---|---|
| Graph | one, fixed across the dataset | fresh per sequence |
| Specified how | implicit | token prefix the model must read |
| Sequence length | 60 | 91 (11 prefix + 80 data) |

## Result 1 — the fixed-graph setting is not measurable

Trained model, best head: **AUROC 0.768** at separating parent from non-parent
positions.

Position-only baselines on the same scoring set:

| Scorer | AUROC |
|---|---|
| Offset-only (parent rate by distance, no model) | **0.968** |
| Recency-only (`score = -distance`) | **0.926** |
| Trained model, best head | 0.768 |

The model scores *below* a scorer with no parameters. With one fixed graph,
parents always sit at the same offsets, so "recover the graph" and "memorise two
offsets" are the same task and AUROC cannot separate them.

The ablation is similarly weak. Total headroom between using the structure and
not using it is `log K − H(ε) ≈ 1.386` nats. Single-head parent ablation moved
loss by **0.085**, about **6%** of that. That indicates redundancy across heads
and the MLP, not — as an earlier draft of this README claimed — a single head
carrying the task.

The control was also too generous: ablating randomly chosen non-parent cells
removes near-zero attention mass, so its ~0.0002 effect was close to a tautology.

## Result 2 — fixing the data is not enough; the metric is the problem

Giving each sequence its own graph (`data_incontext.py`) should kill the
positional confound. It does not, on its own:

| Scorer, in-context task, all allowed cells | AUROC |
|---|---|
| Offset-only | 0.946 |
| Recency-only | 0.946 |

Parents are always *recent* — within the last ~`d` tokens — while the negative
set is dominated by far-away cells no attention pattern would visit. Recency
alone still nearly solves it.

The fix is to score only against the candidate set the parent is actually drawn
from: the `d` cells of the previous timestep block (`lag_candidate_pairs`).

| Scorer, restricted candidate set | AUROC |
|---|---|
| Offset-only | 0.535 |
| Recency-only | 0.486 |

Now position carries no information and anything above 0.5 has to come from
reading the prefix. `analyze.py` recomputes both baselines on every run and
prints a verdict; an AUROC that does not clear them is not reported as a result.

## Result 3 — in-context structure learning

Trained on the in-context task (8,000 steps, ~7 min on CPU, 411,520 parameters).

| Metric | Value | Reference |
|---|---|---|
| Accuracy, predictable positions | **0.9591** | ceiling 0.9600 |
| Val loss, all targets | **0.4046** | all-target ceiling 0.4003 (**+1.1%**) |
| AUROC, best head (restricted set) | **0.8478** | offset-only 0.5373, recency-only 0.4863 |
| Parent ablation | loss 1.9525 | **+1.548 nats**, above uniform (log 5 = 1.609) |
| Count-matched control | loss 0.4045 | **−0.0001 nats** |

**The model is essentially Bayes-optimal.** Accuracy sits 0.0009 below the
ceiling and loss 1.1% above it. Note that the all-target ceiling (0.4003) is the
right comparison, not the predictable-only figure (0.2234): the loss is averaged
over every target, including the prefix and the `t=0` block, which are
unpredictable by construction.

**Attention clears the position baselines by a wide margin** — 0.848 against
0.537, on a scoring set where position carries no information.

**The structure is confined to layer 1:**

```
             head0   head1   head2   head3
layer 0     0.4977  0.5028  0.4947  0.5029     <- all at chance
layer 1     0.6621  0.8478  0.5769  0.6563
```

Every first-layer head is at chance while the second layer carries the signal.
The routing is not available from the raw embedding: layer 0 has to move prefix
information into position before layer 1 can act on it. That is a composition
signature, and it is the most interesting thing in the project.

**Ablation.** Suppressing the best head's attention to true parent positions
raises loss by 1.548 nats, to 1.9525 — *above* the uniform-prediction level of
`log 5 = 1.609`. Removing the head does not merely leave the model uninformed,
it leaves it confidently wrong.

### Caveat on the control

The control removes the same **count** of cells, not the same attention mass.
Mass matching was attempted and is not achievable here: the head places ~0.77 of
its mass on the single parent cell and ~0.016 across the other four candidates,
so the parent ablation removes 57.8 units of mass per sequence while the entire
non-parent candidate pool holds only 1.19. There is no comparable mass to match
against.

That the control cannot be constructed is itself a measurement of how sharp the
head is, but it does mean the near-zero control delta is weaker evidence than a
true mass-matched comparison would be. A wrong-parent ablation — same count,
same structural role, different identity — is the stronger control and is the
next thing to run.

## Layout

```
model.py             attention, block, transformer, plus five correctness checks
data.py              fixed-graph generator (Result 1)
data_incontext.py    per-sequence graph, prefix encoding, restricted scoring set
train.py             training loop for either task
analyze.py           AUROC, position baselines, mass-matched ablation, figures
```

## Running it

```bash
python model.py                                  # correctness checks 1-5
python data_incontext.py                         # generator sanity output
python train.py --task incontext --steps 8000
python analyze.py --ckpt model_incontext.pt --figures
```

Minutes on a laptop; the model is ~400k parameters.

## Correctness checks

`model.py` runs five: output shapes, attention rows summing to one, the causal
mask (both a triangularity assert and a future-perturbation test), agreement
with `F.scaled_dot_product_attention` as a reference oracle, and overfitting a
single batch to near-zero loss.

## Two things that are easy to get wrong

**The off-by-one.** The head at position `p-1` predicts the token at position
`p`, so the attention row scored against `adjacency[p]` is row `p-1`. Getting
this wrong yields maps that look nearly right and an AUROC at chance.

**Unreachable parents.** Instantaneous parents of the first variable in a
timestep sit at or after `p-1` and are masked out by construction, so they are
excluded from scoring.

## Scope

This is a controlled synthetic setting. It says nothing directly about attention
in language models trained on natural data — the ground truth exists here
precisely because the data was constructed. What it provides is a testbed where
an interpretability claim can be checked against a known answer and against
baselines strong enough to falsify it.

## Open

- wrong-parent ablation as a stronger control than count matching
- ablate depth (1 vs 2 layers) and positional encoding scheme
- vary graph density and `ε`; find where recovery degrades
- prefix-free variant: infer the graph purely from the observed prefix of the
  sequence, with no explicit encoding