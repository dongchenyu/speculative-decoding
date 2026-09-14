import torch
import torch.nn as nn

class EagleDraft(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, hidden, token_embedding):
        # hidden: [B, hidden_dim]
        # token_embedding: [B, embed_dim]
        
        x = torch.cat([hidden, token_embedding], dim=-1)
        next_hidden = self.net(x)
        
        return next_hidden
    
