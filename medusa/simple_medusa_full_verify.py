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
    return torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x, attention_mask):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = precompute_rope(self.head_dim, S, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~attention_mask.unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )
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
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        
    def forward(self, x, attention_mask):
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class TinyTreeTransformerLM(nn.Module):
    def __init__(self, vocab_size=1000, dim=128, num_heads=4, num_layers=2, ffn_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        return logits, hidden
    
def build_toy_tree():
    return [
        {"name": "A",     "token": 101, "parent": None},
        {"name": "B",     "token": 102, "parent": None},
        {"name": "A->C",  "token": 201, "parent": 0},
        {"name": "A->D",  "token": 202, "parent": 0},
        {"name": "B->C",  "token": 201, "parent": 1},
        {"name": "B->D",  "token": 202, "parent": 1},
    ]
    
def get_ancestors(nodes, node_idx):
    path = []
    cur = node_idx
    while cur is not None:
        path.append(cur)
        cur = nodes[cur]["parent"]
    path.reverse()
    return path

def get_leaf_indices(nodes):
    parent_set = set()
    for node in nodes:
        if node["parent"] is not None:
            parent_set.add(node["parent"])
            
    return [idx for idx in range(len(nodes)) if idx not in parent_set]

def get_candidate_paths(nodes):
    return [get_ancestors(nodes, leaf_idx) for leaf_idx in get_leaf_indices(nodes)]
 
def build_tree_attention_mask(nodes):
    N = len(nodes)
    mask = torch.zeros(N, N, dtype=torch.bool)
    for i in range(N):
        for j in get_ancestors(nodes, i):
            mask[i, j] = True
    return mask

def build_full_attention_mask(prefix_len, tree_mask, device):
    T = prefix_len
    N = tree_mask.shape[0]
    total = T + N
    
    full_mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    
    for i in range(T):
        full_mask[i, :i + 1] = True
        
    full_mask[T:, :T] = True
    full_mask[T:, T:] = tree_mask.to(device)
    
    return full_mask

def build_tree_input(prefix, nodes):
    tree_tokens = torch.tensor(
        [[node["token"] for node in nodes]],
        device=prefix.device,
        dtype=prefix.dtype,
    )
    return torch.cat([prefix, tree_tokens], dim=1)

def extract_target_predictions(logits, prefix_len, nodes):
    target_predictions = {}
    
    root_pred = torch.argmax(
        logits[:, prefix_len - 1, :],
        dim=-1,
    ).item()
    
    target_predictions["root"] = root_pred
    
    for i in range(len(nodes)):
        node_position = prefix_len + i
        
        node_pred = torch.argmax(
            logits[:, node_position, :],
            dim=-1,
        ).item()
        
        target_predictions[i] = node_pred
        
    return target_predictions

def verify_one_path(nodes, path, target_predictions):
    accepted = 0
    
    first_idx = path[0]
    first_token = nodes[first_idx]["token"]
    
    if first_token != target_predictions["root"]:
        return accepted
    
    accepted += 1
    
    for depth in range(1, len(path)):
        parent_idx = path[depth - 1]
        node_idx = path[depth]
        
        candidate_token = nodes[node_idx]["token"]
        target_pred = target_predictions[parent_idx]
        
        if candidate_token != target_pred:
            break
        
        accepted += 1
        
    return accepted

def verify_all_paths(nodes, target_predictions):
    best_path = None
    best_accepted = -1
    
    print("\n========== verify paths ==========")
    
    for path in get_candidate_paths(nodes):
        names = [nodes[idx]["name"] for idx in path]
        
        accepted = verify_one_path(
            nodes,
            path,
            target_predictions,
        )
        
        print(f"path={names}, accepted={accepted}")
        
        if accepted > best_accepted:
            best_accepted = accepted
            best_path = path
            
    return best_path, best_accepted

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTreeTransformerLM().to(device)
    
    prefix = torch.tensor(
        [[10, 20, 30, 40]],
        device=device,
    )
    
    nodes = build_toy_tree()
    
    tree_mask = build_tree_attention_mask(nodes)
    full_mask = build_full_attention_mask(
        prefix_len=prefix.shape[1],
        tree_mask=tree_mask,
        device=device,
    )
    
    full_input = build_tree_input(prefix, nodes)
    
    print("prefix:", prefix.tolist())
    print("\nfull input:", full_input.tolist())
    print("\nlayout:")
    print("[10,20,30,40 | A,B,A->C,A->D,B->C,B->D]")
    print("\nattention mask shape:", full_mask.shape)
    
    logits, hidden = model(
        full_input,
        full_mask,
    )
    
    print("\nlogits shape:", logits.shape)
    print("hidden shape:", hidden.shape)
    
    target_predictions = extract_target_predictions(
        logits,
        prefix_len=prefix.shape[1],
        nodes=nodes,
    )
    
    print("\n========== target predictions from real logits ==========")
    print("root ->", target_predictions["root"])
    
    for i, node in enumerate(nodes):
        print(
            f"state {node['name']:<5} "
            f"(node {i}) -> next token "
            f"{target_predictions[i]}"
        )
        
    best_path, best_accepted = verify_all_paths(
        nodes,
        target_predictions,
    )
    
    best_names = [nodes[idx]["name"] for idx in best_path]
    accepted_tokens = [
        nodes[idx]["token"]
        for idx in best_path[:best_accepted]
    ]
    
    print("\n========== best accepted path ==========")
    print("best path:", best_names)
    print("accepted length:", best_accepted)
    print("accepted tokens:", accepted_tokens)
    
    print("\nNOTE:")
    print(
        "The Transformer and tree tokens are random/untrained, "
        "so accepted length can be 0. "
        "The point is the dataflow: "
        "tree -> tree mask -> one target forward "
        "-> per-state logits -> verify paths."
    )
    
if __name__ == "__main__":
    main()