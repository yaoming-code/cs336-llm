import torch
import numpy as np
import torch.nn as nn
import math
from einops import einsum, rearrange, reduce
from typing import Optional
from collections.abc import Callable, Iterable

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        sigma = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
    def forward(self, x):
        return einsum(x, self.W, "... i, o i -> ... o")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.embedding = torch.nn.init.trunc_normal_(self.embedding, 0, 1, a=-3, b=3)
    def forward(self, token_ids: torch.tensor) -> torch.tensor:
        return self.embedding[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x: torch.tensor) -> torch.tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(reduce(x**2 , "b s d -> b s 1", 'mean'))
        x = x / RMS
        x = einsum(x, self.weight, "b s d, d -> b s d")
        result = x.to(in_dtype)
        return result
    
class SWiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))
    def forward(self, x: torch.tensor) -> torch.tensor:
        x1 = einsum(x, self.W1, "b s dm, df dm -> b s df")
        x3 = einsum(x, self.W3, "b s dm, df dm -> b s df")
        x = torch.sigmoid(x1) * x1
        x = x * x3
        x = einsum(self.W2, x, "dm df, b s df -> b s dm")
        return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        positions = torch.arange(0, max_seq_len, 1)
        freq = 1 / theta ** ((torch.arange(0, d_k, 2) ).float() / d_k)
        freqs = torch.outer(positions, freq)
        self.register_buffer("cos_cache", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cache", torch.sin(freqs), persistent=False)
    def forward(self, x: torch.tensor, token_positions:torch.tensor) -> torch.tensor:
        cos_pos = self.cos_cache[token_positions]
        sin_pos = self.sin_cache[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_new_even = x_even * cos_pos - x_odd * sin_pos
        x_new_odd = x_even * sin_pos + x_odd * cos_pos
        out = torch.empty_like(x)
        out[..., 0::2] = x_new_even
        out[..., 1::2] = x_new_odd
        return out
    
def softmax(x: torch.tensor, i: int) -> torch.tensor:
    c = torch.max(x, dim=i, keepdim=True)[0]
    x = x - c
    exp_v = torch.exp(x)
    return exp_v / torch.sum(exp_v, dim=i, keepdim=True)

def scaled_dot_product_attention(query: torch.tensor, key: torch.tensor, value: torch.tensor, mask=None) -> torch.tensor:
    d_k = query.size(-1)
    scaled_attension_scores = einsum(query, key, "... s_q d_k, ... s_k d_k -> ... s_q s_k") / math.sqrt(d_k)
    if mask != None:
        scaled_attension_scores = scaled_attension_scores.masked_fill(~mask, float('-inf'))
    scaled_attension_scores = softmax(scaled_attension_scores, -1)
    return einsum(scaled_attension_scores, value, "... s_q s_k, ... s_k d_v -> ... s_q d_v")

class multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model: int, heads: int, max_seq_len: int, theta=10000.0):
        super().__init__()
        self.wq = nn.Parameter(torch.empty(d_model, d_model))
        self.wk = nn.Parameter(torch.empty(d_model, d_model))
        self.wv = nn.Parameter(torch.empty(d_model, d_model))
        self.wo = nn.Parameter(torch.empty(d_model, d_model))
        sigma = math.sqrt(2/(2 * d_model))
        nn.init.trunc_normal_(self.wq, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.wk, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.wv, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.wo, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.heads= heads
        self.rope = RotaryPositionalEmbedding(theta, d_model/heads, max_seq_len)
    def forward(self, x: torch.tensor, token_positions:torch.tensor) ->torch.tensor:
        q = einsum(x, self.wq, "... n d_in, d_out d_in -> ... n d_out")
        k = einsum(x, self.wk, "... n d_in, d_out d_in -> ... n d_out")
        v = einsum(x, self.wv, "... n d_in, d_out d_in -> ... n d_out")
        q = rearrange(q, "... n (h d_k) -> ... h n d_k", h=self.heads)
        k = rearrange(k, "... n (h d_k) -> ... h n d_k", h=self.heads)
        v = rearrange(v, "... n (h d_v) -> ... h n d_v", h=self.heads)
        ##token_positions = token_positions.unsqueeze(1)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        a = torch.ones(q.size(-2), q.size(-2))
        mask = torch.tril(a, diagonal=0).bool()
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... h n d_v -> ... n h d_v")
        out = rearrange(out, "... n h d_v -> ... n (h d_v)")
        result = out @ self.wo.T
        return result

class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.wq = nn.Parameter(torch.empty(d_model, d_model))
        self.wk = nn.Parameter(torch.empty(d_model, d_model))
        self.wv = nn.Parameter(torch.empty(d_model, d_model))
        self.wo = nn.Parameter(torch.empty(d_model, d_model))
        sigma = math.sqrt(2/(2 * d_model))
        nn.init.trunc_normal_(self.wq, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.wk, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.wv, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.wo, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.heads = heads
    def forward(self, x: torch.tensor) ->torch.tensor:
        q = einsum(self.wq, x, "d_out d_in, ... n d_in -> ... n d_out")
        k = einsum(self.wk, x, "d_out d_in, ... n d_in -> ... n d_out")
        v = einsum(self.wv, x, "d_out d_in, ... n d_in -> ... n d_out")
        q = rearrange(q, "... n (h d_k) -> ... h n d_k", h=self.heads)
        k = rearrange(k, "... n (h d_k) -> ... h n d_k", h=self.heads)
        v = rearrange(v, "... n (h d_v) -> ... h n d_v", h=self.heads)
        a = torch.ones(q.size(-2), q.size(-2))
        mask = torch.tril(a, diagonal=0).bool()
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "... h n d_m -> ... n (h d_m)")
        out = out @ self.wo.T
        return out

class transformer_block(nn.Module):
    def __init__(self, d_model: int, heads: int, dff: int, max_seq_len: int, theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dff = dff
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.multihead_self_attention_with_rope = multihead_self_attention_with_rope(d_model, heads, max_seq_len, theta)
        self.ff = SWiGLU(d_model, dff)
    def forward(self, x: torch.tensor,) -> torch.tensor:
        x_norm = self.norm1(x)
        token_positions = torch.arange(x.size(-2), device=x.device)
        out = x + self.multihead_self_attention_with_rope(x_norm, token_positions)
        out = out + self.ff(self.norm2(out))
        return out

class transformer_lm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, context_length, vocab_size, num_layers, rope_theta=10000.0):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.layers_transformer_block = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers_transformer_block: 
            x = layer(x)
        x = self.norm(x)
        return self.linear(x)
        
def cross_entropy(logits: torch.tensor, output: torch.tensor) -> torch.tensor:
    c = torch.max(logits, dim=-1, keepdim=True)[0]
    shifted = logits - c
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    target = logits.gather(dim=-1, index=output.unsqueeze(-1))
    loss = c + log_sum_exp - target
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.99), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group["params"]:
                if(p.grad is None):
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m = state.get("m")
                v = state.get("v")
                t = state.get("t", 0)
                t = t + 1
                grad = p.grad.data
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state['t'] = t
        return loss
def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return t / T_w * alpha_max
    elif t <= T_c:
        return alpha_min + (alpha_max - alpha_min) / 2 * (1 + math.cos(math.pi * ((t - T_w) / (T_c - T_w))))
    else:
        return alpha_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], 
                      max_norm: float, 
                      eps: float = 1e-6) -> None:
    grads = [g.grad for g in params if g.grad != None]
    if (len(grads) == 0):
        return
    grads_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    if(grads_norm > max_norm):
        scaling = max_norm / (grads_norm + eps)
        for g in grads:
            g *= scaling

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    n = len(x)
    starts = np.random.randint(0, n - context_length, size = batch_size)
    input = np.stack([x[i : i + context_length] for i in starts])
    output = np.stack([x[i + 1 : i + 1 + context_length] for i in starts])
    input = torch.from_numpy(input).long().to(device)
    output = torch.from_numpy(output).long().to(device)
    return input, output

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state" : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "t": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    t = checkpoint["t"]
    return t

def decoding(model, prompt_ids, max_new_tokens, temperature, top_p, eot_id):
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long).squeeze(-1)
    with torch.no_grad:
        for _ in range(max_new_tokens):
            logits = model(x)[..., -1, ...]
            logits = logits / temperature
            probs = softmax(logits, -1)
            probs = top_p(probs, top_p)
            next_id = torch.multinomial(probs, samples=1)
            x = torch.cat([x, next_id], dim=0)
            if (next_id.item() == eot_id):
                break
    return x.squeeze(0).tolist()

def top_k_filter(probs, p):
    sorted_probs, sorted_ids = torch.sort(probs, dim=-1)
    consums = torch.consum(sorted_probs, dim=-1)
    mask = consums - sorted_probs < p
    sorted_probs = sorted_probs * mask
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    new_probs = torch.zeros_like(sorted_probs)
    new_probs.scatter_(-1, sorted_ids, sorted_probs)
    return new_probs