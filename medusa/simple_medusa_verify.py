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

def get_leaf_indices(nodes):
    parent_set = set()
    
    for node in nodes:
        if node["parent"] is not None:
            parent_set.add(node["parent"])
            
    leaves = []
    
    for idx in range(len(nodes)):
        if idx not in parent_set:
            leaves.append(idx)
            
    return leaves

def get_candidate_paths(nodes):
    leaves = get_leaf_indices(nodes)
    paths = []
    
    for leaf_idx in leaves:
        paths.append(get_ancestors(nodes, leaf_idx))
        
    return paths

def build_fake_target_predictions():
    return {
        "root": 101,
        0: 202,
        1: 201,
    }
    
def verify_one_path(nodes, path, target_predictions):
    accepted = 0
    
    # 第一个 candidate token 和 root prediction 比较
    first_idx = path[0]
    first_token = nodes[first_idx]["token"]
    
    if first_token != target_predictions["root"]:
        return accepted
    
    accepted += 1
    
    # 后续 candidate token: 用父节点对应状态的 target prediction 来验证
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
    paths = get_candidate_paths(nodes)
    
    best_path = None
    best_accepted = -1
    
    print("========== verify candidate paths ==========")
    
    for path in paths:
        names = [nodes[idx]["name"] for idx in path]
        accepted = verify_one_path(nodes, path, target_predictions)
        
        print(
            f"path={names}, "
            f"accepted={accepted}"
        )
        
        if accepted > best_accepted:
            best_accepted = accepted
            best_path = path
            
    return best_path, best_accepted

def print_tree(nodes):
    print("========== tree ==========")

    for idx, node in enumerate(nodes):
        print(
            f"{idx}: "
            f"name={node['name']:<5} "
            f"token={node['token']} "
            f"parent={node['parent']}"
        )
        
def print_target_predictions(target_predictions):
    print("\n========== fake target predictions ==========")
    print("root    ->", target_predictions["root"], "(A)")
    print("after A ->", target_predictions[0], "(D)")
    print("after B ->", target_predictions[1], "(C)")
    
def main():
    nodes = build_toy_tree()
    target_predictions = build_fake_target_predictions()
    
    print_tree(nodes)
    print_target_predictions(target_predictions)
    
    best_path, best_accepted = verify_all_paths(nodes, target_predictions)
    
    best_names = [
        nodes[idx]["name"]
        for idx in best_path
    ]
    
    best_tokens = [
        nodes[idx]["token"]
        for idx in best_path[:best_accepted]
    ]
    
    print("\n========== best accepted path ==========")
    print("best path:", best_names)
    print("accepted length:", best_accepted)
    print("accepted tokens:", best_tokens)
        
if __name__ == "__main__":
    main()