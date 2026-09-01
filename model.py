"""
Decoder-only transformer, written from scratch.

You implement every TODO. Do not import nn.Transformer, nn.MultiheadAttention,
or F.scaled_dot_product_attention in the model itself -- the whole point of the
project is that you wrote the attention. (The test at the bottom uses SDPA as a
reference oracle, which is fine and is exactly how you'd check it in practice.)

Run `python src/model.py` at any point. It runs five correctness checks. Get all
five passing before you go near the training loop; every hour spent debugging a
training run that was actually a broken mask is an hour wasted.

Target size: d_model=128, n_heads=4, n_layers=2, d_ff=512, ~200k params.
Small on purpose -- 8 attention maps total, all of them readable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    forward(x) -> (out, attn)
        x    : [B, L, d_model]
        out  : [B, L, d_model]
        attn : [B, n_heads, L, L]   post-softmax weights, kept for analysis

    Returning attn is not optional here -- analyze.py scores these maps against
    the ground-truth graph, so make sure they survive the forward pass.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # TODO: q, k, v projections and the output projection.
        #       One fused nn.Linear(d_model, 3*d_model) is idiomatic; three
        #       separate ones are easier to read. Either is fine.
        # TODO: dropout module if you want it.
        raise NotImplementedError

    def forward(self, x):
        B, L, _ = x.shape
        # TODO: project to q, k, v
        # TODO: reshape to [B, n_heads, L, d_head]
        # TODO: scores = q @ k^T / sqrt(d_head)          -> [B, H, L, L]
        # TODO: causal mask -- position i may attend to j <= i only.
        #       torch.triu(torch.ones(L, L), diagonal=1).bool() marks the
        #       forbidden cells; masked_fill them with -inf BEFORE softmax.
        #       Build the mask on x.device or you will get a device error the
        #       first time you touch a GPU.
        # TODO: attn = softmax(scores, dim=-1)
        # TODO: out = attn @ v, merge heads, output projection
        raise NotImplementedError


class Block(nn.Module):
    """Pre-norm block: x = x + attn(ln1(x)); x = x + mlp(ln2(x)).

    Pre-norm, not post-norm. Post-norm needs learning-rate warmup to train
    stably at this depth and you do not need that headache.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # TODO: ln1, attn, ln2, mlp (Linear -> GELU -> Linear)
        raise NotImplementedError

    def forward(self, x):
        # TODO: return (x, attn) -- pass the attention map up so the top-level
        #       model can collect one per layer
        raise NotImplementedError


class CausalTransformer(nn.Module):
    """
    forward(idx) -> (logits, attns)
        idx    : [B, L] int64 token ids
        logits : [B, L, vocab_size]
        attns  : list of [B, n_heads, L, L], one per layer
    """

    def __init__(self, vocab_size: int, seq_len: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        # TODO: token embedding [vocab_size, d_model]
        # TODO: learned positional embedding [seq_len, d_model]
        #       Learned, not sinusoidal. Position IS the variable identity here
        #       (position p = t*d + i), so you want the model free to learn
        #       structure over it. Worth an ablation later.
        # TODO: stack of Blocks, final layernorm, lm_head to vocab
        raise NotImplementedError

    def forward(self, idx):
        # TODO: embed tokens + positions, run blocks collecting attn maps,
        #       final ln, project to logits
        raise NotImplementedError

    def loss(self, idx):
        """Next-token cross-entropy. Predict idx[:, 1:] from idx[:, :-1].

        Mind this off-by-one everywhere. The head at position p-1 predicts the
        token at position p, so when analyze.py asks which positions a
        prediction attended to, it must read attention ROW p-1, not row p.
        Getting this wrong produces attention maps that look almost right and
        an AUROC around chance, and it will cost you a full evening.
        """
        logits, _ = self(idx)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            idx[:, 1:].reshape(-1),
        )


# --------------------------------------------------------------------------
# Correctness checks. All five must pass.
# --------------------------------------------------------------------------

def _checks():
    torch.manual_seed(0)
    V, L, B = 5, 60, 8
    m = CausalTransformer(vocab_size=V, seq_len=L)
    idx = torch.randint(0, V, (B, L))

    # 1. shapes
    logits, attns = m(idx)
    assert logits.shape == (B, L, V), logits.shape
    assert len(attns) == len(m.blocks), "one attention map per layer"
    assert attns[0].shape == (B, m.blocks[0].attn.n_heads, L, L)
    print("1. shapes                 OK")

    # 2. attention rows are distributions
    a = attns[0]
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    assert (a >= 0).all()
    print("2. rows sum to 1          OK")

    # 3. causal mask: strictly upper triangle must be exactly zero, and
    #    perturbing the future must not change the present
    assert a.triu(diagonal=1).abs().max() < 1e-9, "attends to the future"
    idx2 = idx.clone()
    idx2[:, L // 2:] = torch.randint(0, V, (B, L - L // 2))
    l2, _ = m(idx2)
    assert torch.allclose(logits[:, :L // 2], l2[:, :L // 2], atol=1e-5), \
        "past predictions changed when the future changed -- mask is leaking"
    print("3. causality              OK")

    # 4. attention matches SDPA as a reference oracle
    blk = m.blocks[0].attn
    x = torch.randn(B, L, blk.d_model)
    out, _ = blk(x)
    #    Re-derive q,k,v the same way your forward does, then compare.
    #    TODO: adapt these three lines to however you named your projections.
    raise NotImplementedError(
        "check 4: wire this up to your own q/k/v projections, then compare "
        "against F.scaled_dot_product_attention(q, k, v, is_causal=True) "
        "followed by your output projection. Should match to ~1e-5."
    )

    # 5. can it overfit a single batch to near-zero loss?
    #    If this fails, the architecture or the optimiser is wrong. Never debug
    #    a full training run before this passes.
    m2 = CausalTransformer(vocab_size=V, seq_len=L)
    opt = torch.optim.AdamW(m2.parameters(), lr=3e-3)
    batch = torch.randint(0, V, (4, L))
    for _ in range(400):
        opt.zero_grad()
        loss = m2.loss(batch)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"cannot overfit one batch, loss={loss.item():.4f}"
    print(f"5. overfits one batch     OK (loss {loss.item():.4f})")

    n = sum(p.numel() for p in m.parameters())
    print(f"\nparameters: {n:,}")


if __name__ == "__main__":
    _checks()
