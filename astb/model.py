"""
A minimal decoder-only (GPT-style) causal transformer, trained from scratch.

We train from scratch rather than fine-tuning a pretrained checkpoint because
this sandbox has no network access to model hubs (see README for the
HF-backed version meant for CodeLlama/Llama-3/Mistral on a GPU machine).
Training from scratch is actually a *harder* setting for the attacker than
fine-tuning a capable pretrained model, so any ASR we observe here is a
conservative (lower-bound-ish) signal that the mechanism works at all -- not
an upper bound on real-world attack strength.
"""

import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, T, dh)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        out = att @ v  # (B, nh, T, dh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=3,
                 d_ff=256, max_len=64, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab)

    @torch.no_grad()
    def logprobs_for_prefix(self, idx):
        """Returns log-softmax over vocab at every position given idx as prefix."""
        logits = self.forward(idx)
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id, temperature=0.0, rng=None):
        """Autoregressive generation. temperature=0 -> greedy (argmax)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self.forward(idx_cond)[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if rng is not None:
                    next_id = torch.multinomial(probs, num_samples=1, generator=rng)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if idx.shape[0] == 1 and next_id.item() == eos_id:
                break
        return idx
