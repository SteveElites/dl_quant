"""
Decoder-only transformer, written from scratch.

Target size: d_model=128, n_heads=4, n_layers=2, d_ff=512, ~200k params.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention.

    forward(x) -> (out, attn)
        x    : [B, L, d_model]
        out  : [B, L, d_model]
        attn : [B, n_heads, L, L]   post-softmax weights, kept for analysis
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Fused q, k, v projection
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        B, L, D = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask. NEG_INF is a large finite value rather than -inf: the
        # ablation path adds a second negative bias on top of this, and if a row
        # ever ends up fully suppressed, -inf everywhere makes softmax return
        # NaN while a finite value degrades gracefully to a uniform row.
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, NEG_INF)

        # attn_bias is used for ablation: [B, H, L, L] or broadcastable
        if attn_bias is not None:
            scores = scores + attn_bias

        attn = F.softmax(scores, dim=-1)
        attn_dropped = self.attn_dropout(attn)
        out = attn_dropped @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.resid_dropout(self.c_proj(out))
        return out, attn


class Block(nn.Module):
    """Pre-norm block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None):
        attn_out, attn_map = self.attn(self.ln1(x), attn_bias=attn_bias)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_map


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
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, attn_biases=None):
        B, L = idx.shape
        assert L <= self.seq_len, f"Sequence length {L} exceeds max seq_len {self.seq_len}"

        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)  # [1, L]
        tok_emb = self.token_embedding(idx)                       # [B, L, d_model]
        pos_emb = self.position_embedding(pos)                   # [1, L, d_model]

        x = self.dropout(tok_emb + pos_emb)

        attns = []
        for i, block in enumerate(self.blocks):
            bias = attn_biases[i] if attn_biases is not None else None
            x, attn = block(x, attn_bias=bias)
            attns.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, attns

    def loss(self, idx):
        """Next-token cross-entropy. Predict idx[:, 1:] from idx[:, :-1]."""
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
    
    # Re-derive q, k, v using fused c_attn and shape for SDPA
    q, k, v = blk.c_attn(x).split(blk.d_model, dim=2)
    q = q.view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
    k = k.view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)
    v = v.view(B, L, blk.n_heads, blk.d_head).transpose(1, 2)

    sdpa_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    sdpa_out = sdpa_out.transpose(1, 2).contiguous().view(B, L, blk.d_model)
    expected_out = blk.resid_dropout(blk.c_proj(sdpa_out))

    assert torch.allclose(out, expected_out, atol=1e-5), "attention output does not match SDPA reference"
    print("4. matches SDPA oracle    OK")

    # 5. can it overfit a single batch to near-zero loss?
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