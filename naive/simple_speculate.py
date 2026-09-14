import copy
import torch
from naive.simple_transformer_kvcache import TinyTransformerLM

@torch.no_grad()
def greedy_next_token(model, tokens):
    logits, _ = model(tokens, past_key_values=None)
    return torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)

@torch.no_grad()
def generate_target_greedy(model, input_ids, max_new_tokens=12):
    tokens = input_ids
    for _ in range(max_new_tokens):
        next_token = greedy_next_token(model, tokens)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens

@torch.no_grad()
def draft_tokens(draft_model, prefix, num_draft_tokens):
    tokens = prefix
    proposed = []
    for _ in range(num_draft_tokens):
        next_token = greedy_next_token(draft_model, tokens)
        proposed.append(next_token)
        tokens = torch.cat([tokens, next_token], dim=1)
    return torch.cat(proposed, dim=1)

@torch.no_grad()
def verify_draft_greedy(target_model, prefix, draft):
    B, L = prefix.shape
    K = draft.shape[1]
    assert B == 1
    
    full = torch.cat([prefix, draft], dim=1)
    logits, _ = target_model(full, past_key_values=None)
    
    # logits[:, L-1] 预测 draft[:, 0]
    # logits[:, L]   预测 draft[:, 1]
    # ...
    verify_logits = logits[:, L - 1 : L - 1 + K, :]
    target_predictions = torch.argmax(verify_logits, dim=-1)
    
    accepted = 0
    for i in range(K):
        if target_predictions[0, i].item() == draft[0, i].item():
            accepted += 1
        else:
            break
        
    if accepted < K:
        correction_token = target_predictions[:, accepted:accepted + 1]
        return accepted, False, correction_token, None, target_predictions
    
    bonus_logits = logits[:, L + K - 1, :]
    bonus_token = torch.argmax(bonus_logits, dim=-1, keepdim=True)
    return K, True, None, bonus_token, target_predictions

@torch.no_grad()
def generate_speculative_greedy(
    target_model,
    draft_model,
    input_ids,
    max_new_tokens=12,
    num_draft_tokens=4,
    debug=True
):
    tokens = input_ids
    generated = 0
    round_id = 0
    
    while generated < max_new_tokens:
        round_id += 1
        remain = max_new_tokens - generated
        K = min(num_draft_tokens, remain)
        
        draft = draft_tokens(draft_model, tokens, K)
        accepted, all_accepted, correction, bonus, target_predictions = verify_draft_greedy(target_model, tokens, draft)
        
        if debug:
            print(f"\n========== round {round_id} ==========")
            print("prefix:", tokens.tolist())
            print("draft :", draft.tolist())
            print("target predictions:", target_predictions.tolist())
            print("accepted:", accepted)
    
        if accepted > 0:
            can_take = min(accepted, max_new_tokens - generated)
            tokens = torch.cat([tokens, draft[:, :can_take]], dim=1)
            generated += can_take
        
        if not all_accepted:
            if generated < max_new_tokens:
                tokens = torch.cat([tokens, correction], dim=1)
                generated += 1
                if debug:
                    print("mismatch -> target correction:", correction.tolist())
                
        else:
            if generated < max_new_tokens:
                tokens = torch.cat([tokens, bonus], dim=1)
                generated += 1
                if debug:
                    print("all accepted -> bonus:", bonus.tolist())
    return tokens

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    target_model = TinyTransformerLM(
        vocab_size=1000,
        dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=128
    ).to(device)
    
    draft_model = copy.deepcopy(target_model)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in draft_model.parameters():
            p.add_(0.02 * torch.randn_like(p))
            
    input_ids = torch.tensor([[10, 20, 30, 40]], device=device)
    max_new_tokens = 12
    
    print("========== target greedy baseline ==========")
    baseline = generate_target_greedy(target_model, input_ids, max_new_tokens=max_new_tokens)
    print(baseline)
    
    print("\n========== speculative greedy ==========")
    speculative = generate_speculative_greedy(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=4,
        debug=True
    )
    
    print("\nspeculative output:")
    print(speculative)
    print("\noutputs equal:")
    print(torch.equal(baseline, speculative))

if __name__ == "__main__":
    main()

    