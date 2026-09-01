import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time

# -------- hyperparameters (adjust as you like) ----------
BATCH_SIZE = 64
STEPS = 5000
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_INTERVAL = 100
# ---------------------------------------------------------

# Load data (use the same seed as before)
data = make_dataset(d=5, K=5, T=12, n_train=20000, n_val=2000,
                    eps=0.05, n_inst=1, n_lag=1, seed=0)
train_tokens = torch.tensor(data['train'], dtype=torch.long, device=DEVICE)
val_tokens   = torch.tensor(data['val'],   dtype=torch.long, device=DEVICE)
cfg = data['config']
vocab_size = cfg['vocab_size']
seq_len = cfg['seq_len']
ceiling = data['ceiling']

# Model
model = CausalTransformer(vocab_size, seq_len).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS)

# Training loop
model.train()
for step in range(1, STEPS + 1):
    # sample a batch
    idx = torch.randint(0, len(train_tokens), (BATCH_SIZE,))
    batch = train_tokens[idx]
    loss = model.loss(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    if step % LOG_INTERVAL == 0 or step == 1:
        # ---- validation ----
        model.eval()
        val_loss = 0.0
        correct_total = 0
        n_total = 0
        with torch.no_grad():
            for i in range(0, len(val_tokens), BATCH_SIZE):
                b = val_tokens[i:i+BATCH_SIZE]
                logits, _ = model(b)
                loss_b = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab_size),
                    b[:, 1:].reshape(-1)
                )
                val_loss += loss_b.item() * b.size(0)
                pred = logits[:, :-1].argmax(dim=-1)
                correct_total += (pred == b[:, 1:]).sum().item()
                n_total += b.size(0) * (b.size(1) - 1)
        val_loss /= n_total
        acc = correct_total / n_total

        # accuracy on predictable positions only
        L = val_tokens.size(1)
        p_positions = torch.arange(1, L, device=DEVICE)  # target positions
        mask = (p_positions // cfg['d']) > 0             # predictable = t > 0
        # expand to batch
        mask_expanded = mask.unsqueeze(0).expand(BATCH_SIZE, -1)
        # We need to compute per batch, but we can just compute over all batches
        # We'll compute separately for simplicity:
        correct_pred = 0
        total_pred = 0
        with torch.no_grad():
            for i in range(0, len(val_tokens), BATCH_SIZE):
                b = val_tokens[i:i+BATCH_SIZE]
                logits, _ = model(b)
                pred = logits[:, :-1].argmax(dim=-1)
                targets = b[:, 1:]
                # mask for predictable positions
                # Need to ensure mask shape matches [B, L-1]
                # p_positions is [L-1], we can expand to batch and apply
                mask_batch = mask.unsqueeze(0).expand(b.size(0), -1)
                correct_pred += (pred == targets)[mask_batch].sum().item()
                total_pred += mask_batch.sum().item()
        acc_predictable = correct_pred / total_pred if total_pred > 0 else 0.0

        print(f"step {step:5d} | loss {loss.item():.4f} | val_loss {val_loss:.4f} | "
              f"acc {acc:.4f} | acc_predictable {acc_predictable:.4f} "
              f"(ceiling overall {ceiling['overall']:.4f}, pred-only {ceiling['predictable_only']:.4f})")
        model.train()

# Save the model
torch.save(model.state_dict(), "model.pt")
print("Training complete, model saved as model.pt")