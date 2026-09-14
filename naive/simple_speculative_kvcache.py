import copy
import math
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
    assert head_dim % 2 == 0

    half_dim = head_dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    # [max_seq_len, Dh/2]
    angles = torch.outer(positions, inv_freq)

    cos = angles.cos()
    sin = angles.sin()
    return cos, sin

def apply_rope(x, cos, sin):
    """
    x:   [B, H, S, Dh]
    cos: [S, Dh/2]
    sin: [S, Dh/2]
    """

    half_dim = x.shape[-1] // 2

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,S,Dh/2]
    sin = sin.unsqueeze(0).unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)

def build_causal_mask(past_len, new_len, device):
    total_len = past_len + new_len
    q_pos = torch.arange(past_len, past_len + new_len, device=device).unsqueeze(1)
    k_pos = torch.arange(total_len, device=device).unsqueeze(0)
    return k_pos > q_pos
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x, past_kv=None, position_ids=None):
        B, S_new, D = x.shape
        q = self.q_proj(x).view(B, S_new, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S_new, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S_new, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos_all, sin_all = precompute_rope(self.head_dim, self.max_seq_len, x.device)
        q = apply_rope(q, cos_all[position_ids], sin_all[position_ids])
        k = apply_rope(k, cos_all[position_ids], sin_all[position_ids])
        
        if past_kv is None:
            past_len = 0
            full_k, full_v = k, v
        else:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            full_k = torch.cat([past_k, k], dim=2)
            full_v = torch.cat([past_v, v], dim=2)
            
        scores = q @ full_k.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        mask = build_causal_mask(past_len, S_new, x.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ full_v
        out = out.transpose(1, 2).contiguous().view(B, S_new, D)

        return self.o_proj(out), (full_k, full_v)           
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
class TransformerBlock(nn.Module):
    def __init__(self, dim,num_heads,ffn_dim,max_seq_len):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
    def forward(self, x, past_kv=None, position_ids=None):
        attn_out, new_kv = self.attn(self.attn_norm(x), past_kv, position_ids)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv
    
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=256, max_seq_len=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, ffn_dim, max_seq_len) for _ in range(num_layers)])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
    def forward(self, input_ids, past_key_values=None):
        B, S_new = input_ids.shape
        x = self.token_embedding(input_ids)
        if past_key_values is None:
            past_len = 0
            past_key_values = [None] * len(self.layers)
        else:
            past_len = past_key_values[0][0].shape[2]
        position_ids = torch.arange(past_len, past_len + S_new, device=input_ids.device)
        new_past = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, past_key_values[i], position_ids)
            new_past.append(kv)
        x = self.final_norm(x)
        return self.lm_head(x), new_past
    
def truncate_past_key_values(past_key_values, keep_len):
    return [(k[:,:,:keep_len,:], v[:,:,:keep_len,:]) for k,v in past_key_values]

def cache_len(past_key_values):
    return 0 if past_key_values is None else past_key_values[0][0].shape[2]

@torch.no_grad()
def generate_target_greedy(model, input_ids, max_new_tokens=12):
    tokens = input_ids
    for _ in range(max_new_tokens):
        logits, _ = model(tokens, None)
        nxt = torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
    return tokens

@torch.no_grad()
def draft_tokens(draft_model, prefix, num_draft_tokens):
    tokens = prefix
    proposed = []
    for _ in range(num_draft_tokens):
        logits, _ = draft_model(tokens, None)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        proposed.append(nxt)
        tokens = torch.cat([tokens, nxt], dim=1)
    return torch.cat(proposed, dim=1)

@torch.no_grad()
def init_target_state(target_model, input_ids):
    if input_ids.shape[1] == 1:
        return None, inputs[:, -1:]
    cache_input = input_ids[:, :-1]
    pending = input_ids[:, -1:]
    _, past = target_model(cache_input, None)
    
    return past, pending

@torch.no_grad()
def verify_with_cache(target_model, past_key_values, pending, draft):
    verify_input = torch.cat([pending, draft], dim=1)
    logits, temp_past = target_model(verify_input, past_key_values)
    K = draft.shape[1]
    target_predictions = torch.argmax(logits[:, :K, :],dim=-1)
    accepted = 0
    for i in range(K):
        if target_predictions[0, i].item() == draft[0, i].item():
            accepted += 1
        else:
            break
    if accepted < K:
        correction = target_predictions[:,accepted:accepted + 1]
        return accepted, False, correction, None, target_predictions, temp_past
    bonus = torch.argmax(logits[:, K, :], dim=-1, keepdim=True)
    return K, True, None, bonus, target_predictions, temp_past

@torch.no_grad()
def generate_speculative_kvcache(target_model, draft_model, input_ids, max_new_tokens=12, num_draft_tokens=4, debug=True):
    tokens = input_ids
    past_key_values, pending = init_target_state(target_model, tokens)
    generated = 0
    round_id = 0
    
    while generated < max_new_tokens:
        round_id += 1
        remain = max_new_tokens - generated
        if remain == 1:
            logits, _ = target_model(pending, past_key_values)
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            generated += 1
            break

        K = min(num_draft_tokens, remain - 1)
        draft = draft_tokens(draft_model, tokens, K)
        accepted, all_accepted, correction, bonus, target_preds, temp_past = verify_with_cache(
            target_model, past_key_values, pending, draft)
    
        old_len = cache_len(past_key_values)
        
        if debug:
            print(f"\n========== round {round_id} ==========")
            print("tokens:", tokens.tolist())
            print("cache_len:", old_len)
            print("pending:", pending.tolist())
            print("draft:", draft.tolist())
            print("target predictions:", target_preds.tolist())
            print("accepted:", accepted)
            
        if not all_accepted:
            keep_len = old_len + 1 + accepted
            past_key_values = truncate_past_key_values(temp_past, keep_len)
            
            if accepted > 0:
                tokens = torch.cat([tokens, draft[:, :accepted]], dim=1)
                generated += accepted
                
            if generated < max_new_tokens:
                tokens = torch.cat([tokens, correction], dim=1)
                generated += 1
                pending = correction
            
            if debug:
                print("mismatch -> correction:", correction.tolist())
                print("new cache_len:", cache_len(past_key_values))
                print("new pending:", pending.tolist()) 
                
        else:
            past_key_values = temp_past
            tokens = torch.cat([tokens, draft], dim=1)
            generated += K
            
            if generated < max_new_tokens:
                tokens = torch.cat([tokens, bonus], dim=1)
                generated += 1
                pending = bonus
                
            if debug:
                print("all accepted -> bonus:", bonus.tolist())
                print("new cache_len:", cache_len(past_key_values))
                print("new pending:", pending.tolist())
    return tokens

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = TinyTransformerLM().to(device)
    draft = copy.deepcopy(target)
    
    torch.manual_seed(1)
    with torch.no_grad():
        for p in draft.parameters():
            p.add(0.02 * torch.randn_like(p))
            
    input_ids = torch.tensor([[10,20,30,40]], device=device)
    baseline = generate_target_greedy(target, input_ids, 12)
    print("========== target greedy baseline ==========")
    print(baseline)
    
    speculative = generate_speculative_kvcache(
        target, draft, input_ids,
        max_new_tokens=12,
        num_draft_tokens=4,
        debug=True
    )
    
    print("\n========== speculative + kv cache ==========")
    print(speculative)
    print("\noutputs equal:")
    print(torch.equal(baseline, speculative))
                
if __name__ == "__main__":
    main()