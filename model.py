"""
nanoGPT-plus — architecture modernisée (style Llama / Mistral / Qwen)

Améliorations par rapport au GPT-2 original de nanoGPT :
  1. RMSNorm            : remplace LayerNorm (plus simple, plus rapide, aussi stable)
  2. RoPE               : rotary position embeddings, remplace la table de positions
                          apprise (wpe) — meilleure généralisation, zéro paramètre
  3. SwiGLU             : remplace le MLP GELU — meilleure qualité à budget égal
  4. GQA                : grouped-query attention (n_kv_head < n_head) — réduit la
                          mémoire du KV cache, comme Llama 3 / Mistral
  5. QK-norm            : RMSNorm sur les têtes q et k — stabilise l'entraînement
                          (utilisé par Qwen3, OLMo 2)
  6. KV cache           : génération O(1) par token au lieu de O(T) — sample.py rapide
  7. Flash attention    : via F.scaled_dot_product_attention (déjà dans nanoGPT)
  8. Sans biais partout : moins de paramètres, standard moderne

NOTE : cette architecture est incompatible avec les poids GPT-2 d'OpenAI.
Pour fine-tuner à partir de GPT-2 pré-entraîné, utiliser model_gpt2.py
(l'original de Karpathy) via `--arch=gpt2` dans train.py.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # padding du vocab GPT-2 (50257) au multiple de 64 le plus proche
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 0       # 0 => égal à n_head (pas de GQA) ; ex: 4 têtes kv pour 12 têtes q
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False       # conservé pour compatibilité de config ; ignoré (jamais de biais)
    qk_norm: bool = True     # RMSNorm sur q et k
    rope_theta: float = 10000.0
    arch: str = 'modern'


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019) — sans centrage ni biais."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def precompute_rope(head_dim, max_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin, offset=0):
    # x : (B, n_head, T, head_dim)
    T = x.size(2)
    cos = cos[offset:offset + T].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head > 0 else config.n_head
        assert self.n_head % self.n_kv_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else nn.Identity()
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        offset = 0
        if cache is not None and cache['k'] is not None:
            offset = cache['k'].size(2)
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:
            if cache['k'] is not None:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v

        if self.n_kv_head != self.n_head:  # GQA : chaque tête kv sert n_head/n_kv_head têtes q
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # flash attention ; masque causal seulement quand q et k ont la même longueur
        # (en décodage avec cache, q = 1 token qui doit voir tout le passé)
        is_causal = q.size(2) == k.size(2) and q.size(2) > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))


class SwiGLU(nn.Module):
    """MLP à gating multiplicatif (Shazeer, 2020). hidden ≈ 2/3 de 4*n_embd
    pour garder le même nombre de paramètres que le MLP GELU 4x."""

    def __init__(self, config):
        super().__init__()
        hidden = int(2 * 4 * config.n_embd / 3)
        hidden = 8 * ((hidden + 7) // 8)  # arrondi au multiple de 8 (efficacité GPU)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        cos, sin = precompute_rope(config.n_embd // config.n_head,
                                   config.block_size, config.rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_weights)
        # init réduite pour les projections résiduelles (comme GPT-2)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w3.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        # RoPE n'a aucun paramètre et wte est lié à lm_head : rien à soustraire
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None):
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        x = self.drop(self.wte(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope_cos, self.rope_sin,
                      None if caches is None else caches[i])
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # inférence : dernier token seulement
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.rope_cos = self.rope_cos[:block_size]
        self.rope_sin = self.rope_sin[:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError(
            "L'architecture moderne est incompatible avec les poids GPT-2 d'OpenAI. "
            "Pour fine-tuner depuis GPT-2, utilisez model_gpt2.py (option --arch=gpt2)."
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimation du Model Flops Utilization par rapport aux bfloat16 peak FLOPS d'un A100."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Génération avec KV cache : chaque nouveau token ne recalcule que lui-même,
        au lieu de re-traiter toute la séquence (O(1) vs O(T) par token)."""
        caches = [{'k': None, 'v': None} for _ in self.blocks]
        idx_cond = idx if idx.size(1) <= self.config.block_size \
            else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond, caches=caches)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if caches[0]['k'] is not None and caches[0]['k'].size(2) >= self.config.block_size:
                # fenêtre pleine : on repart d'un cache neuf sur les derniers block_size tokens
                caches = [{'k': None, 'v': None} for _ in self.blocks]
                logits, _ = self(idx[:, -self.config.block_size:], caches=caches)
            else:
                logits, _ = self(idx_next, caches=caches)
        return idx
