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
        # x: [B, S, D]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
def precompute_rope(head_dim, max_seq_len, device):
    assert head_dim % 2 == 0
    
    half_dim = head_dim // 2
    
    # [Dh / 2]
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    
    # [S]
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    
    # [S, Dh / 2]
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
    
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    # [1, 1, S, Dh/2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    return torch.cat([out1, out2], dim=-1)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        """
        x: [B, S, D]
        """
        
        B, S, D = x.shape
        
        # [B, S, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # [B, H, S, Dh]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = precompute_rope(self.head_dim, S, x.device)
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        
        casual_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(casual_mask, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        
        # [B,H,S,Dh]
        out = torch.matmul(attn, v)
        
        # [B,S,D]
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, S, D)
        
        out = self.o_proj(out)
        
        return out
    
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        """
        x: [B, S, D]
        """
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        hidden = F.silu(gate) * up
        
        out = self.down_proj(hidden)
        
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, max_seq_len):
        super().__init__()
        
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, max_seq_len)
        
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        
    def forward(self, x):
        # Attention block
        x = x + self.attn(self.attn_norm(x))
        
        # FFN block
        x = x + self.ffn(self.ffn_norm(x))
        
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=256, max_seq_len=128):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ]) 
        
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, input_ids):
        """
        input_ids: [B, S]
        """
        
        # x: [B, S, D]
        x = self.token_embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        
        logits = self.lm_head(x)
        
        return logits

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=8):
    tokens = input_ids
    
    for _ in range(max_new_tokens):
        logits = model(tokens)
        
        next_token_logits = logits[:, -1, :]
        
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        tokens = torch.cat([tokens, next_token], dim=1)
        
    return tokens

def main():
    torch.manual_seed(0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TinyTransformerLM(vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=256, max_seq_len=128).to(device)
    
    inputs_ids = torch.tensor([[10, 20, 30, 40],], device=device)
    
    print("inputs_ids:")
    print(inputs_ids)
    print("shape:", inputs_ids.shape)
    
    logits = model(inputs_ids)
    
    print("\nlogits shape:")
    print(logits.shape)
    
    output = generate(model, inputs_ids, max_new_tokens=5)
    
    print("\ngenerated tokens:")
    print(output)
    
if __name__ == "__main__":
    main()
    
        
    