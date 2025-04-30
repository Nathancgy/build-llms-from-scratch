import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    """ Model configuration """
    def __init__(self, vocab_size=10000, block_size=256, n_layer=6, n_head=8, n_embd=512, dropout=0.1, eps=1e-5, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.eps = eps # Epsilon for numerical stability in division
        for k, v in kwargs.items():
            setattr(self, k, v)

class LinearCausalSelfAttention(nn.Module):
    """ Linear Causal Self-Attention using ELU activation function """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.eps = config.eps

    def phi(self, x):
        # Feature map function based on the blog post: elu(x) + 1
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, C = x.size() # Batch, Sequence length, Embedding dim
        head_size = C // self.n_head

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        # Apply feature map
        phi_q = self.phi(q) # (B, nh, T, hs)
        phi_k = self.phi(k) # (B, nh, T, hs)

        # Calculate cumulative sums for causal calculation (scan operation)
        # Outer product phi_k * v^T summed up causally
        kv_state = torch.einsum('bnhk,bnhv->bnhkv', phi_k, v) # (B, nh, T, hs_k, hs_v) - assuming hs_k=hs_v=head_size
        S = torch.cumsum(kv_state, dim=2) # (B, nh, T, hs_k, hs_v) - Cumulative state matrix

        # Cumulative sum for the normalization factor
        Z = torch.cumsum(phi_k, dim=2) # (B, nh, T, hs_k) - Cumulative normalization state

        # Calculate query-state products for numerator and denominator
        # Numerator: phi_q * S (dot product along hs_k dimension)
        numerator = torch.einsum('bnhk,bnhkv->bnhv', phi_q, S) # (B, nh, T, hs_v)

        # Denominator: phi_q * Z (dot product along hs_k dimension)
        denominator = torch.einsum('bnhk,bnhk->bnh', phi_q, Z) # (B, nh, T)
        # Add epsilon for stability and unsqueeze for broadcasting
        denominator = denominator.unsqueeze(-1).clamp(min=self.eps) # (B, nh, T, 1)

        # Calculate final output
        y = numerator / denominator # (B, nh, T, hs_v)

        # Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """ Position-wise Feed-Forward Network """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    """ Transformer Decoder Block (Pre-LN) using Linear Attention """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = LinearCausalSelfAttention(config) # Use Linear Attention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Residual connection after attention
        x = x + self.mlp(self.ln_2(x))  # Residual connection after FFN
        return x

class GPT(nn.Module):
    """ GPT Language Model with Linear Attention """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Positional embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]), # Decoder blocks
            ln_f = nn.LayerNorm(config.n_embd), # Final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Output head

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
             torch.nn.init.zeros_(module.bias)
             torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)

        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x) # (b, t, n_embd)
        logits = self.lm_head(x) # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # NOTE: This generation method still recalculates the full sequence attention
        # at each step. For optimal linear attention generation speed, one would
        # implement a version that carries forward the KV state (S and Z).
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if needed, but linear attention *could* handle longer
            # sequences if memory allows state accumulation. Sticking to block_size limit here.
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # Focus on last token, apply temperature

            if top_k is not None: # Apply top-k filtering
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx