import torch
from EAGLE.eagle_transformer import EagleTargetWrapper
from EAGLE.eagle_draft import EagleDraft
    
@torch.no_grad()
def eagle_generate(target, eagle_draft, input_ids, max_steps=3):
    device = input_ids.device
    
    candidates = []
    
    # hidden: [B,S,D] -> [B, D]
    logits, hidden = target(input_ids, return_hidden=True)
    hidden = hidden[:, -1, :]
    
    next_logits = logits[:, -1, :]
    
    token = torch.argmax(next_logits, dim=-1)
    candidates.append(token)
    
    for _ in range(max_steps - 1):
        token_embed = target.token_embedding(token)
        hidden = eagle_draft(hidden, token_embed)
        
        logits = target.lm_head(hidden)
        token = torch.argmax(logits, dim=-1)
        
        candidates.append(token)
        
    return torch.stack(candidates, dim=1)
    
@torch.no_grad()
def verify_candidates(target, input_ids, candidates):
    # input_ids: [1, prefix_len]
    # candidates: [1, num_candidates]
    
    assert input_ids.shape[0] == 1
    assert candidates.shape[0] == 1
    
    # input_ids: A B C 
    # candidates: D E F
    prefix_len = input_ids.shape[1]
    num_candidates = candidates.shape[1]
    
    verify_ids = torch.cat([input_ids, candidates], dim=1)
    
    logits = target(verify_ids)
    
    verify_logits = logits[:, prefix_len - 1 : prefix_len - 1 + num_candidates, :]
    target_tokens = torch.argmax(verify_logits, dim=-1)
    
    accepted = 0
    
    for i in range(num_candidates):
        if(candidates[0,i] == target_tokens[0,i]):
            accepted += 1
        else:
            break
        
    return accepted, target_tokens 
    
@torch.no_grad()
def eagle_verify_step(target, input_ids, candidates):
    accepted, target_tokens = verify_candidates(target, input_ids, candidates)
    num_candidates = candidates.shape[1]
    accepted_tokens = candidates[:, :accepted]
    
    if accepted < num_candidates:
        correction_token = target_tokens[:, accepted:accepted + 1]
        output = torch.cat([input_ids, accepted_tokens, correction_token], dim=1)
        
        return output, accepted

    output = torch.cat([input_ids, candidates], dim=1)
    return output, accepted

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    target = EagleTargetWrapper(
        vocab_size=1000,
        dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=128,
    ).to(device)
    
    draft = EagleDraft(hidden_dim=128, embed_dim=128).to(device)
    
    input_ids = torch.tensor([[10, 20, 30, 40]], device=device)
    
    candidates = eagle_generate(target, draft, input_ids, max_steps=3)
    
    print("prefix:")
    print(input_ids)
    
    print("\ndraft candidates:")
    print(candidates)
    
    accepted, target_tokens = verify_candidates(target, input_ids, candidates)
    
    print("\ntarget verify tokens:")
    print(target_tokens)
    
    print("\naccepted:")
    print(accepted)
    
    output, accepted = eagle_verify_step(target, input_ids, candidates)
    
    print("\nfinal output:")
    print(output)
    
if __name__ == "__main__":
    main()
    
    