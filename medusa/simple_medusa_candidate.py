import torch
import torch.nn as nn
import torch.nn.functional as F

# Medusa Head 类似 LM Head
# hidden_state -> logits
class MedusaHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        
    def forward(self, hidden):
        logits = self.proj(hidden)
        return logits
    
# Multiple Medusa Heads
class MedusaHeads(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            MedusaHead(hidden_dim, vocab_size)
            for _ in range(num_heads)
        ])
        
    def forward(self, hidden):
        outputs = []
        for head in self.heads:
            logits = head(hidden)
            outputs.append(logits)
        return outputs
    
# top-k candidate: 
# logits: [B, vocab]
# return: token ids
def get_topk_candidates(logits, k):
    probs = torch.softmax(logits, dim=-1)
    values, tokens = torch.topk(probs, k, dim=-1)
    return tokens, values
    
def build_candidate_tree(head_candidates):
    nodes = []
    parents = [
        None
        for _ in range(len(head_candidates[0]))
    ]
    
    for token in head_candidates[0]:
        nodes.append({
            "token": token,
            "parent": None,
        })
        
    current_level = list(range(len(nodes)))
    
    for depth in range(1, len(head_candidates)):
        next_level = []
        for parent_idx in current_level:
            for token in head_candidates[depth]:
                node_idx = len(nodes)
                nodes.append({
                    "token": token,
                    "parent": parent_idx,
                })
                
                next_level.append(node_idx)
                
        current_level = next_level
        
    return nodes

def print_tree(nodes):
    print("\n========== candidate tree ==========")
    
    for idx, node in enumerate(nodes):
        print(
            f"node {idx}: "
            f"token={node['token']} "
            f"parent={node['parent']}"
        )
        
def main():
    torch.manual_seed(0)
    
    hidden_dim = 128
    vocab_size = 1000

    num_heads = 3

    topk = 2

    # 假装一个Transformer
    hidden = torch.randn(1, hidden_dim)
    
    print("hidden shape:", hidden.shape)
    
    medusa = MedusaHeads(hidden_dim, vocab_size, num_heads)
    logits_list = medusa(hidden)
    
    print("\n========== head logits ==========")
    
    for i, logits in enumerate(logits_list):
        print(f"head {i}:", logits.shape)
        
    all_candidates = []
    
    print("\n========== head candidates ==========")
    
    for i, logits in enumerate(logits_list):
        tokens, probs = get_topk_candidates(logits, topk)
        
        tokens = tokens[0]
        probs = probs[0]
        
        print(f"\nhead {i}:")
        print("tokens:", tokens.tolist())
        print("probs:", probs.tolist())
        
        all_candidates.append(tokens.tolist())
        
    nodes = build_candidate_tree(all_candidates)
    print_tree(nodes)
    
if __name__ == "__main__":
    main()
    