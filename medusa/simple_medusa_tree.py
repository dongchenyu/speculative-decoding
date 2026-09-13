import torch

def build_toy_tree():
    nodes = [
        {"name": "A",     "token": 101, "parent": None},
        {"name": "B",     "token": 102, "parent": None},
        {"name": "A->C",  "token": 201, "parent": 0},
        {"name": "A->D",  "token": 202, "parent": 0},
        {"name": "B->C",  "token": 201, "parent": 1},
        {"name": "B->D",  "token": 202, "parent": 1},
    ]
    return nodes

def get_ancestors(nodes, node_idx):
    path = []
    cur = node_idx
    
    while cur is not None:
        path.append(cur)
        cur = nodes[cur]["parent"]
        
    path.reverse()
    return path

def build_tree_attention_mask(nodes):
    N = len(nodes)
    mask = torch.zeros(N, N, dtype=torch.bool)
    
    for i in range(N):
        ancestors = get_ancestors(nodes, i)
        for j in ancestors:
            mask[i, j] = True
            
    return mask

def print_tree(nodes):
    print("========== tree nodes ==========")
    
    for idx, node in enumerate(nodes):
        print(
            f"{idx}: "
            f"name={node['name']:<5} "
            f"token={node['token']} "
            f"parent={node['parent']}"
        )

def build_full_attention_mask(prefix_len, tree_mask):
    T = prefix_len
    N = tree_mask.shape[0]
    total = T + N
    
    full_mask = torch.zeros(total, total, dtype=torch.bool)
    
    # prefix -> prefix: standard causal attention
    for i in range(T):
        for j in range(i + 1):
            full_mask[i, j] = True
    
    # tree node -> all prefix tokens
    full_mask[T:, :T] = True
    
    # tree node -> only its own ancestors/self inside tree
    full_mask[T:, T:] = tree_mask

    return full_mask    
     
def print_paths(nodes):
    print("\n========== ancestor paths ==========")
    
    for idx, node in enumerate(nodes):
        path = get_ancestors(nodes, idx)
        names = [nodes[p]["name"] for p in path]
        
        print(
            f"{idx}: {node['name']:<5} "
            f"-> {names}"
        )
        
def print_mask(mask, row_names, col_names):
    print("      ", end="")
    
    for name in col_names:
        print(f"{name:>6}", end="")
    
    print()
    
    for i, row_name in enumerate(row_names):
        print(f"{row_name:>6}", end="")
    
        for j in range(mask.shape[1]):
            value = 1 if mask[i, j] else 0
            print(f"{value:>6}", end="")
    
        print()
        
def main():
    nodes = build_toy_tree()
    
    print_tree(nodes)
    print_paths(nodes)
    
    tree_mask = build_tree_attention_mask(nodes)
    names = [node["name"] for node in nodes]
    
    print("\n========== tree attention mask ==========")
    print("1 = can attend, 0 = cannot attend")
    
    print_mask(tree_mask, names, names)
    
    prefix_len = 4
    
    full_mask = build_full_attention_mask(prefix_len, tree_mask)
    
    full_names = [
        "p0",
        "p1",
        "p2",
        "p3",
    ] + names
    
    print("\n========== full attention mask ==========")
    print("sequence layout:")
    print("[p0 p1 p2 p3 A B A->C A->D B->C B->D]")
    
    print_mask(full_mask, full_names, full_names)
    
if __name__ == "__main__":
    main()