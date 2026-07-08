"""
Benchmark : nanoGPT baseline (GPT-2 style) vs nanoGPT optimisé (style Llama/Mistral)
Améliorations testées : RMSNorm, RoPE (rotary embeddings), SwiGLU
Entraînement identique sur shakespeare_char, CPU.
"""
import os, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_num_threads(os.cpu_count())

# ----------------- config commune -----------------
n_layer, n_head, n_embd = 4, 4, 128
block_size, batch_size = 64, 12
max_iters, eval_interval, eval_iters = 300, 50, 20
learning_rate, weight_decay = 1e-3, 0.1
vocab_size = 65
device = 'cpu'

data_dir = os.path.join('data', 'shakespeare_char')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split, rng):
    data = train_data if split == 'train' else val_data
    ix = rng.integers(len(data) - block_size, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x, y

# ----------------- modèle OPTIMISÉ -----------------
class RMSNorm(nn.Module):
    """Remplace LayerNorm : plus simple, plus rapide, aussi efficace (Llama, Mistral)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def precompute_rope(head_dim, max_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd)
    T = x.size(2)
    cos, sin = cos[:T].unsqueeze(0).unsqueeze(0), sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

class OptAttention(nn.Module):
    """Attention avec RoPE : encode les positions par rotation de q/k
    (meilleure généralisation aux longueurs, pas de table de positions apprise)."""
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        cos, sin = precompute_rope(n_embd // n_head, block_size)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        q, k = apply_rope(q, self.cos, self.sin), apply_rope(k, self.cos, self.sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    """Remplace le MLP GELU : gating multiplicatif, meilleure qualité à budget égal.
    hidden dimensionné à ~2/3 de 4*n_embd pour garder le même nombre de paramètres."""
    def __init__(self):
        super().__init__()
        hidden = int(2 * 4 * n_embd / 3)
        hidden = 8 * ((hidden + 7) // 8)  # arrondi multiple de 8
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, n_embd, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class OptBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.attn = RMSNorm(n_embd), OptAttention()
        self.n2, self.mlp = RMSNorm(n_embd), SwiGLU()
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class OptGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)  # pas de wpe : RoPE s'en charge
        self.blocks = nn.ModuleList([OptBlock() for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying (comme nanoGPT)
        self.apply(self._init)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('w3.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

# ----------------- entraînement identique pour les deux -----------------
def make_optimizer(model):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': nodecay, 'weight_decay': 0.0}],
        lr=learning_rate, betas=(0.9, 0.95))

@torch.no_grad()
def eval_loss(model, rng):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch('val', rng)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()

def train(model, name):
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)       # mêmes batches pour les deux modèles
    eval_rng = np.random.default_rng(42)
    opt = make_optimizer(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} — {n_params:,} paramètres ===")
    history, t0 = [], time.time()
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            vl = eval_loss(model, np.random.default_rng(42))
            history.append((it, vl))
            print(f"  iter {it:4d} | val loss {vl:.4f} | {time.time()-t0:6.1f}s")
        x, y = get_batch('train', rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    dt = time.time() - t0
    toks = max_iters * batch_size * block_size
    print(f"  → total {dt:.1f}s | {toks/dt:,.0f} tokens/s | val loss finale {history[-1][1]:.4f}")
    return history, dt, n_params

if __name__ == '__main__':
    from model import GPTConfig, GPT
    base_cfg = GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer,
                         n_head=n_head, n_embd=n_embd, dropout=0.0, bias=False)
    torch.manual_seed(1337)
    baseline = GPT(base_cfg)
    h1, t1, p1 = train(baseline, "BASELINE nanoGPT (GPT-2 : LayerNorm + pos. apprises + GELU)")

    torch.manual_seed(1337)
    optimized = OptGPT()
    h2, t2, p2 = train(optimized, "OPTIMISÉ (RMSNorm + RoPE + SwiGLU, style Llama)")

    print("\n" + "=" * 62)
    print(f"{'iter':>6} | {'baseline':>10} | {'optimisé':>10} | {'gain':>8}")
    for (it, a), (_, b) in zip(h1, h2):
        print(f"{it:>6} | {a:>10.4f} | {b:>10.4f} | {a-b:>+8.4f}")
    print("=" * 62)
    print(f"Paramètres : baseline {p1:,} vs optimisé {p2:,}")
    print(f"Vitesse    : baseline {t1:.1f}s vs optimisé {t2:.1f}s")
