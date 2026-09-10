import copy
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
def precompute_rope(head_dim, max_seq_len, device):
    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = precompute_rope(self.head_dim, S, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        scores = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim))
        mask = torch.triu(torch.ones(S, S, device=x.device,dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x) * self.gate_proj(x)))
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=256, max_seq_len=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim, max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)
    
def logits_to_probs(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be > 0')
    return F.softmax(logits / temperature, dim=-1)

def sample_from_probs(probs):
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate_target_sampling(model, input_ids, max_new_tokens=12, temperature=1.0):
    tokens = input_ids
    for _ in range(max_new_tokens):
        logits = model(tokens)
        probs = logits_to_probs(logits[:, -1, :], temperature)
        next_token = sample_from_probs(probs)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens

@torch.no_grad()
def draft_tokens_sampling(draft_model, prefix, num_draft_tokens, temperature=1.0):
    tokens = prefix
    proposed = []
    draft_probs = []
    
    for _ in range(num_draft_tokens):
        logits = draft_model(tokens)
        q = logits_to_probs(logits[:, -1, :], temperature)
        next_token = sample_from_probs(q)
        proposed.append(next_token)
        draft_probs.append(q)
        tokens = torch.cat([tokens, next_token], dim=1)
        
    return torch.cat(proposed, dim=1), draft_probs

@torch.no_grad()
def verify_speculative_sampling(target_model, prefix, draft, draft_probs, temperature=1.0, debug=False):
    B, L = prefix.shape
    K = draft.shape[1]
    assert B == 1
    assert len(draft_probs) == K
    
    full = torch.cat([prefix, draft], dim=1)
    logits = target_model(full)
    
    verify_logits = logits[:, L - 1 : L - 1 + K,:]
    target_probs = F.softmax(verify_logits / temperature, dim=-1)
    
    accepted = 0
    eps = 1e-12
    
    for i in range(K):
        token = draft[:, i:i + 1]
        p = target_probs[:,i,:]
        q = draft_probs[i]
        
        p_token = torch.gather(p, 1, token)
        q_token = torch.gather(q, 1, token)
        accept_prob = torch.clamp(p_token / (q_token + eps), max=1.0)
        u = torch.rand_like(accept_prob)
        
        if debug:
            print(
                f'i={i}, token={token.item()}, '
                f'p={p_token.item():.6f}, q={q_token.item():.6f}, '
                f'accept_prob={accept_prob.item():.6f}, u={u.item():.6f}'
            )
            
        if (u < accept_prob).item():
            accepted += 1
            continue
        
        residual = torch.clamp(p - q, min=0.0)
        residual_sum = residual.sum(dim=-1, keepdim=True)
        if(residual_sum <= eps).any():
            residual = p
        else:
            residual = residual / residual_sum
            
        correction = sample_from_probs(residual)
        return accepted, False, correction, None
    
    bonus_probs = logits_to_probs(logits[:, L + K - 1, :], temperature)
    bonus = sample_from_probs(bonus_probs)
    return K, True, None, bonus

@torch.no_grad()
def generate_speculative_sampling(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens=12,
    num_draft_tokens=4,
    temperature=1.0,
    debug=True,
):
    tokens = input_ids
    generated = 0
    round_id = 0
    total_drafted = 0
    total_accepted = 0
    
    while generated < max_new_tokens:
        round_id += 1
        remain = max_new_tokens - generated
        K = min(num_draft_tokens, remain)
        
        draft, draft_probs = draft_tokens_sampling(
            draft_model, tokens, K, temperature
        )
        total_drafted += K
        
        accepted, all_accepted, correction, bonus = verify_speculative_sampling(
            target_model,
            tokens,
            draft,
            draft_probs,
            temperature=temperature,
            debug=False
        )
        total_accepted += accepted
        
        if debug:
            print(f'\n========== round {round_id} ==========')
            print('prefix:', tokens.tolist())
            print('draft :', draft.tolist())
            print('accepted:', accepted)
            
        if accepted > 0:
            can_take = min(accepted, max_new_tokens - generated)
            tokens = torch.cat([tokens, draft[:, :can_take]], dim=1)
            generated += can_take
            if generated >= max_new_tokens:
                break
        
        if not all_accepted:
            if generated < max_new_tokens:
                tokens = torch.cat([tokens, correction], dim=1)
                generated += 1
                if debug:
                    print('reject -> correction:', correction.tolist())
        else:
            if generated < max_new_tokens:
                tokens = torch.cat([tokens, bonus], dim=1)
                generated += 1
                if debug:
                    print('all accepted -> bonus:', bonus.tolist())
    
    if debug:
        rate = total_accepted / total_drafted if total_drafted else 0.0
        print(f'\nacceptance rate: {rate:.4f}')
        
    return tokens
    
@torch.no_grad()
def sample_target_next_token(target_model, prefix, temprature=1.0):
    logits = target_model(prefix)
    probs = logits_to_probs(logits[:, -1, :], temprature)
    return sample_from_probs(probs).item()

@torch.no_grad()
def sample_speculative_first_new_token(
    target_model,
    draft_model,
    prefix,
    num_draft_tokens=4,
    temperature=1.0
):
    draft, draft_probs = draft_tokens_sampling(
        draft_model, prefix, num_draft_tokens, temperature
    )
    
    accepted, _, correction, _ = verify_speculative_sampling(
        target_model,
        prefix,
        draft,
        draft_probs,
        temperature=temperature,
        debug=False
    )
    
    if accepted > 0:
        return draft[0, 0].item()
    return correction[0, 0].item()

def compare_next_token_distributions(
    target_model,
    draft_model,
    prefix,
    num_samples=2000,
    num_draft_tokens=4,
    temperature=1.0,
    top_n=10
):
    target_counter = Counter()
    speculative_counter = Counter()
    
    for _ in range(num_samples):
        target_counter[sample_target_next_token(target_model, prefix, temperature)] += 1
                    
    for _ in range(num_samples):
        speculative_counter[
            sample_speculative_first_new_token(
                target_model,
                draft_model,
                prefix,
                num_draft_tokens,
                temperature
            )
        ] += 1
    
    all_tokens = set(target_counter) | set(speculative_counter)
    rows = []
    for token in all_tokens:
        tf = target_counter[token] / num_samples
        sf = speculative_counter[token] / num_samples
        rows.append((token, tf, sf, abs(tf - sf)))
    
    rows.sort(key=lambda x: max(x[1], x[2]), reverse=True)
    
    print('\n========== empirical next-token distribution ==========')
    print('token | target_freq | speculative_freq | abs_diff')
    for token, tf, sf, diff in rows[:top_n]:
        print(f'{token:5d} | {tf:11.4f} | {sf:16.4f} | {diff:.4f}')
    
def main():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    target_model = TinyTransformerLM().to(device)
    draft_model = copy.deepcopy(target_model)
    
    torch.manual_seed(1)
    with torch.no_grad():
        for p in draft_model.parameters():
            p.add_(0.02 * torch.randn_like(p))
            
    input_ids = torch.tensor([[10, 20, 30, 40]], device=device)
    temperature = 1.0
    
    print('========== one speculative sampling run ==========')
    torch.manual_seed(123)
    output = generate_speculative_sampling(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=12,
        num_draft_tokens=4,
        temperature=temperature,
        debug=True
    )
    
    print('\noutput:')
    print(output)
    
    torch.manual_seed(999)
    compare_next_token_distributions(
        target_model,
        draft_model,
        input_ids,
        num_samples=200,
        num_draft_tokens=4,
        temperature=temperature,
        top_n=10,
    )
    
if __name__ == '__main__':
    main()