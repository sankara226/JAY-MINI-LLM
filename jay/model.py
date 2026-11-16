"""The GPT model, copied from the notebook (STEPs 2b, 11-14).

The settings and layer names must stay exactly as they are:
the saved .pt weights only fit this exact shape.
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Model size (STEP 2b) ---
VOCAB_SIZE = 4096  # number of different tokens the model knows
BLOCK_SIZE = 256   # context window: how many tokens the model sees at once
N_LAYER    = 6     # number of Transformer blocks stacked
N_HEAD     = 6     # number of attention "heads" per block
N_EMBD     = 384   # size of each token's vector (384 / 6 = 64 per head)
DROPOUT    = 0.1   # randomly turns off 10% of neurons during training


class CausalSelfAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)   # makes q, k and v for each token
    self.proj = nn.Linear(N_EMBD, N_EMBD)      # mixes the heads' results together
    self.attn_drop = nn.Dropout(DROPOUT)
    self.resid_drop = nn.Dropout(DROPOUT)
    self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

  def forward(self, x):
    B, T, C = x.shape       # Batch, Time (tokens), Channels (384)
    hs  = C // N_HEAD       # size of each head: 384 / 6 = 64

    q, k, v = self.qkv(x).split(N_EMBD, dim=2)
    q = q.view(B, T, N_HEAD, hs).transpose(1, 2)   # -> (B, heads, T, 64)
    k = k.view(B, T, N_HEAD, hs).transpose(1, 2)
    v = v.view(B, T, N_HEAD, hs).transpose(1, 2)

    scores = q @ k.transpose(-2, -1) / math.sqrt(hs)   # how well each query matches each key
    scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))    # hide the future
    weights = F.softmax(scores, dim=-1)
    self.last_weights = weights.detach()               # kept for attention heatmaps
    weights = self.attn_drop(weights)

    out = weights @ v                                     # collect a weighted mix of values
    out = out.transpose(1, 2).contiguous().view(B, T, C)  # glue the 6 heads back together
    return self.resid_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),   # expand: 384 -> 1536
            nn.GELU(),                       # non-linearity
            nn.Linear(4 * N_EMBD, N_EMBD),   # shrink back: 1536 -> 384
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.ln1 = nn.LayerNorm(N_EMBD)
    self.attn = CausalSelfAttention()
    self.ln2 = nn.LayerNorm(N_EMBD)
    self.mlp = MLP()

  def forward(self, x):
    x = x + self.attn(self.ln1(x))  # communicate
    x = x + self.mlp(self.ln2(x))   # think
    return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)   # token id -> 384 numbers
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)   # position -> 384 numbers
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)  # 384 numbers -> 4096 scores
        self.head.weight = self.tok_emb.weight  # weight tying: input and output share the same table
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, stop_token=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and next_id.item() == stop_token:
                break
        self.train()
        return idx
