import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reload data (must use same seed)
data = make_dataset(d=5, K=5, T=12, n_train=20000, n_val=2000,
                    eps=0.05, n_inst=1, n_lag=1, seed=0)
cfg = data['config']
vocab_size = cfg['vocab_size']
seq_len = cfg['seq_len']
adjacency = data['adjacency']  # numpy bool [L, L]
d = cfg['d']

# Load model
model = CausalTransformer(vocab_size, seq_len).to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

val_tokens = torch.tensor(data['val'], dtype=torch.long, device=DEVICE)

# 1. Compute mean attention over validation set (first 512 samples)
n_samples = min(512, len(val_tokens))
attn_sum = None
with torch.no_grad():
    for i in range(0, n_samples, 128):
        batch = val_tokens[i:i+128]
        _, attns = model(batch)
        stacked = torch.stack([a.mean(0) for a in attns])  # [layers, H, L, L]
        if attn_sum is None:
            attn_sum = stacked
        else:
            attn_sum += stacked
mean_attn = (attn_sum / ((n_samples + 127) // 128)).cpu().numpy()  # [layers, H, L, L]

# 2. Edge-recovery AUROC (per head)
def edge_auroc(attn_maps, adjacency):
    L = adjacency.shape[0]
    n_layers, n_heads = attn_maps.shape[:2]
    per_head = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            scores = []
            labels = []
            for p in range(1, L):
                row = p - 1
                for q in range(row + 1):
                    scores.append(attn_maps[layer, head, row, q])
                    labels.append(adjacency[p, q])
            scores = np.array(scores)
            labels = np.array(labels, dtype=bool)
            if labels.sum() == 0 or (~labels).sum() == 0:
                auroc = np.nan
            else:
                order = np.argsort(-scores)
                y = labels[order]
                tp = np.cumsum(y)
                fp = np.cumsum(~y)
                auroc = np.trapz(tp / tp[-1], fp / fp[-1])   # works with numpy >=1.23
            per_head[layer, head] = auroc
    best = np.nanmax(per_head)
    best_idx = np.unravel_index(np.nanargmax(per_head), per_head.shape)
    return {
        'per_head': per_head,
        'best_head': best,
        'best_head_index': best_idx,
        'mean_head': np.nanmean(per_head),
    }

auroc_res = edge_auroc(mean_attn, adjacency)
print("AUROC per head:")
print(auroc_res['per_head'])
print(f"Best head: {auroc_res['best_head']:.4f} at index {auroc_res['best_head_index']}")
print(f"Mean over heads: {auroc_res['mean_head']:.4f}")

# 3. Ablation for the best head
best_layer, best_head = auroc_res['best_head_index']
print(f"\nRunning ablation on layer {best_layer}, head {best_head}")

# Build parent positions to ablate
L = adjacency.shape[0]
ablate_positions = []
for p in range(1, L):
    row = p - 1
    for q in range(row + 1):
        if adjacency[p, q]:
            ablate_positions.append((row, q))
print(f"Total parent edges to ablate: {len(ablate_positions)}")

# Random ablation: sample same number of non‑parent positions
rng = np.random.default_rng(0)
all_non_parent = []
for p in range(1, L):
    row = p - 1
    for q in range(row + 1):
        if not adjacency[p, q]:
            all_non_parent.append((row, q))
if len(all_non_parent) < len(ablate_positions):
    print("Warning: not enough non-parent positions, using all")
    random_positions = all_non_parent
else:
    indices = rng.choice(len(all_non_parent), size=len(ablate_positions), replace=False)
    random_positions = [all_non_parent[i] for i in indices]

# Create bias tensors (B=1, later expanded inside forward if needed)
def make_bias(positions, head_idx, L, n_heads=4, device='cpu'):
    bias = torch.zeros(1, n_heads, L, L, device=device)
    for row, col in positions:
        bias[:, head_idx, row, col] = float("-inf")
    return bias

bias_parent = make_bias(ablate_positions, best_head, L, device=DEVICE)
bias_random = make_bias(random_positions, best_head, L, device=DEVICE)

# Compute losses on a validation subset
val_subset = val_tokens[:256]
vocab = vocab_size

def loss_with_biases(bias_list):
    # bias_list: list of biases per layer (None for untouched layers)
    with torch.no_grad():
        logits, _ = model(val_subset, attn_biases=bias_list)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab),
            val_subset[:, 1:].reshape(-1)
        ).item()
    return loss

# Base loss (no bias)
base_loss = loss_with_biases([None] * len(model.blocks))

# Parent ablation
biases_parent = [None] * len(model.blocks)
biases_parent[best_layer] = bias_parent
loss_parent = loss_with_biases(biases_parent)

# Random ablation
biases_random = [None] * len(model.blocks)
biases_random[best_layer] = bias_random
loss_random = loss_with_biases(biases_random)

print(f"Base loss:        {base_loss:.4f}")
print(f"Parent ablation:  {loss_parent:.4f}  (Δ = {loss_parent - base_loss:.4f})")
print(f"Random ablation:  {loss_random:.4f}  (Δ = {loss_random - base_loss:.4f})")

# The headline result
print("\n--- Headline ---")
print(f"Parent Δ - Random Δ = {(loss_parent - base_loss) - (loss_random - base_loss):.4f}")