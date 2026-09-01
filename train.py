"""
Training loop. Runs standalone:

    python train.py --task incontext     # fresh graph per sequence (main result)
    python train.py --task fixed         # single fixed graph (baseline / null)
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import CausalTransformer


def load(task: str):
    if task == "incontext":
        from data_incontext import make_dataset
        data = make_dataset(d=5, K=5, T=16, n_train=40000, n_val=4000,
                            eps=0.05, seed=0)
        return (data, data["predictable"],
                data["ceiling"]["accuracy"], data["ceiling"]["loss"])
    from data import make_dataset
    data = make_dataset(d=5, K=5, T=12, n_train=20000, n_val=2000,
                        eps=0.05, n_inst=1, n_lag=1, seed=0)
    d, L = data["config"]["d"], data["config"]["seq_len"]
    predictable = np.array([(c + 1) // d > 0 for c in range(L - 1)])
    return data, predictable, data["ceiling"]["predictable_only"], None


@torch.no_grad()
def evaluate(model, val, predictable, vocab_size, batch=256):
    """One pass. Returns per-token val loss and accuracy on predictable positions.

    Both quantities are accumulated per TOKEN and divided by a token count. The
    earlier version weighted the loss by sequence count and divided by tokens,
    understating it by a factor of the sequence length.
    """
    model.eval()
    loss_sum = n_loss = corr = n_corr = 0
    mask = torch.as_tensor(predictable, device=val.device)
    for i in range(0, len(val), batch):
        b = val[i:i + batch]
        logits, _ = model(b)
        loss_sum += F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                    b[:, 1:].reshape(-1),
                                    reduction="sum").item()
        n_loss += b[:, 1:].numel()
        hit = (logits[:, :-1].argmax(-1) == b[:, 1:])
        m = mask.unsqueeze(0).expand(b.size(0), -1)
        corr += hit[m].sum().item()
        n_corr += int(m.sum().item())
    model.train()
    return loss_sum / n_loss, corr / n_corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["incontext", "fixed"], default="incontext")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=250)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or f"model_{args.task}.pt"

    data, predictable, ceiling_acc, ceiling_loss = load(args.task)
    cfg = data["config"]
    train = torch.as_tensor(data["train"], dtype=torch.long, device=device)
    val = torch.as_tensor(data["val"], dtype=torch.long, device=device)

    model = CausalTransformer(cfg["vocab_size"], cfg["seq_len"],
                              n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    print(f"task {args.task} | device {device} | seq_len {cfg['seq_len']} | "
          f"vocab {cfg['vocab_size']}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"ceiling: acc {ceiling_acc:.4f}"
          + (f", loss {ceiling_loss:.4f}" if ceiling_loss else ""))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=args.steps)
    gen = torch.Generator().manual_seed(args.seed)

    t0, history = time.time(), []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train), (args.batch_size,), generator=gen)
        loss = model.loss(train[idx.to(device)])
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == 1:
            vl, acc = evaluate(model, val, predictable, cfg["vocab_size"])
            history.append((step, loss.item(), vl, acc))
            print(f"step {step:6d} | train {loss.item():.4f} | val {vl:.4f} | "
                  f"acc {acc:.4f} / {ceiling_acc:.4f} | {time.time()-t0:.0f}s")

    torch.save({"state_dict": model.state_dict(),
                "args": vars(args), "history": history}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
