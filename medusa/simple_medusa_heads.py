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
    
        scores = q @ k.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
    
        mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    
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
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)
    
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
    def __init__(
        self,
        vocab_size=1000,
        dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=128,
    ):
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
    
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
    
        return logits, hidden
        
class MedusaHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim,vocab_size,bias=False)
        
    def forward(self, hidden):
        x = F.silu(self.proj(hidden))
        return self.lm_head(x)
    
class TinyMedusaHeads(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_medusa_heads=4):
        super().__init__()
        
        self.heads = nn.ModuleList([
            MedusaHead(hidden_dim, vocab_size)
            for _ in range(num_medusa_heads)
        ])
    
    def forward(self, hidden):
        all_logits = []
        for head in self.heads:
            logits = head(hidden)
            all_logits.append(logits.unsqueeze(1))
        
        return torch.cat(all_logits, dim=1)
    
@torch.no_grad()
def print_medusa_topk(medusa_logits, top_k=3):
    probs = F.softmax(medusa_logits, dim=-1)
    values, indices = torch.topk(probs, k=top_k, dim=-1)

    _, num_heads, _ = medusa_logits.shape

    print("\n========== Medusa heads top-k ==========")

    for head_idx in range(num_heads):
        print(f"\nhead {head_idx}:")

        for rank in range(top_k):
            token = indices[0, head_idx, rank].item()
            prob = values[0, head_idx, rank].item()

            print(
                f"  top{rank + 1}: "
                f"token={token}, "
                f"prob={prob:.6f}"
            )
            
def main():
    torch.manual_seed(0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 1000
    hidden_dim = 128
    num_medusa_heads = 4
    
    target_model = TinyTransformerLM(
        vocab_size=vocab_size,
        dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=128
    ).to(device)
    
    medusa_heads = TinyMedusaHeads(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_medusa_heads=num_medusa_heads
    ).to(device)
    
    input_ids = torch.tensor([[10, 20, 30, 40]],device=device)
    
    print("input_ids:")
    print(input_ids)
    
    target_logits, hidden_states = target_model(input_ids)
    
    print("\ntarget logits shape:", target_logits.shape)
    print("hidden states shape:", hidden_states.shape)
    
    last_hidden = hidden_states[:, -1, :]
    
    print("\nlast hidden shape:", last_hidden.shape)
    
    target_next = torch.argmax(target_logits[:, -1, :], dim=-1)
    
    print("target normal next token:", target_next.tolist())
    
    medusa_logits = medusa_heads(last_hidden)
    
    print("\nmedusa logits shape:", medusa_logits.shape)
    
    print_medusa_topk(medusa_logits, top_k=3)
    
if __name__ == "__main__":
    main()
    
    