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

    def forward(self, x, past_kv=None, position_ids=None):
        """
        x:
            [B, S_new, D]

        past_kv:
            past_k: [B,H,S_past,Dh]
            past_v: [B,H,S_past,Dh]

        position_ids:
            [S_new]
        """

        B, S, D = x.shape

        # 1. QKV projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # [B,S,D] -> [B,H,S,Dh]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. RoPE
        # 关键点：
        # 只给“当前新产生的 Q/K”做它们自己的 position 对应的 RoPE。
        # 旧的 K cache 已经在过去做过 RoPE，绝对不能再次旋转。
        
        cos_all, sin_all = precompute_rope(self.head_dim, self.max_seq_len, x.device)
        
        cos = cos_all[position_ids]
        sin = sin_all[position_ids]
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        new_k = k
        new_v = v
        
        # 3. KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv

            # past_k 已经做过它自己的 RoPE
            k = torch.cat([past_k, new_k], dim=2)
            v = torch.cat([past_v, new_v], dim=2)

        # 这里保存的是“已经做完 RoPE 的 K”和普通 V
        new_kv = (k, v)

        # 4. Attention
        # q: [B,H,S_new,Dh]
        # k: [B,H,S_total,Dh]
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        # 我们当前只处理两种情况：
        #
        # 1) Prefill: past_kv is None
        #    Q_len == K_len == S
        #    需要 causal mask
        #
        # 2) 单 token Decode: S_new == 1
        #    当前 token 可以看全部历史 + 自己
        #    不需要 mask
        if past_kv is None:
            causal_mask = torch.triu(
                torch.ones(
                    S,
                    S,
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )

            scores = scores.masked_fill(
                causal_mask,
                float("-inf"),
            )

        attn = F.softmax(scores, dim=-1)

        # [B,H,S_new,S_total] @ [B,H,S_total,Dh]
        # -> [B,H,S_new,Dh]
        out = torch.matmul(attn, v)

        # merge heads
        # [B,H,S,Dh] -> [B,S,D]
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        out = self.o_proj(out)

        return out, new_kv
        
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = F.silu(gate) * up

        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, max_seq_len):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(self, x, past_kv=None, position_ids=None):
        attn_out, new_kv = self.attn(
            self.attn_norm(x),
            past_kv=past_kv,
            position_ids=position_ids,
        )

        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_kv


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
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, past_key_values=None, debug=False):
        """
        input_ids:
            Prefill: [B,S]
            Decode:  [B,1]

        past_key_values:
            [
                (K_layer0, V_layer0),
                (K_layer1, V_layer1),
                ...
            ]
        """

        B, S = input_ids.shape

        x = self.token_embedding(input_ids)

        # past_len 就是当前 cache 中已经有多少个 token。
        #
        # 在这个 toy 场景中：
        # past_len == 当前新 token 的绝对 position 起点。
        if past_key_values is None:
            past_len = 0
            past_key_values = [None] * len(self.layers)
        else:
            past_len = past_key_values[0][0].shape[2]

        position_ids = torch.arange(
            past_len,
            past_len + S,
            device=input_ids.device,
        )

        if debug:
            print(
                f"input shape={tuple(input_ids.shape)}, "
                f"past_len={past_len}, "
                f"position_ids={position_ids.tolist()}"
            )

        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            x, new_kv = layer(
                x,
                past_kv=past_key_values[i],
                position_ids=position_ids,
            )

            new_past_key_values.append(new_kv)

            if debug and i == 0:
                k, v = new_kv
                print(
                    f"layer0 cache: "
                    f"K={tuple(k.shape)}, "
                    f"V={tuple(v.shape)}"
                )

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_past_key_values


@torch.no_grad()
def generate_naive(model, input_ids, max_new_tokens=5):
    """
    不使用 KV Cache。
    每轮都把完整 tokens 重新送进模型。
    """

    tokens = input_ids

    for _ in range(max_new_tokens):
        logits, _ = model(
            tokens,
            past_key_values=None,
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


@torch.no_grad()
def generate_kvcache(model, input_ids, max_new_tokens=5, debug=False):
    """
    使用 KV Cache：

    1. 第一次完整 prompt 做 prefill
    2. 后续 decode 每轮只输入最新 token
    """

    tokens = input_ids

    # ----------------------------
    # Prefill
    # ----------------------------
    logits, past_key_values = model(
        input_ids,
        past_key_values=None,
        debug=debug,
    )

    next_token = torch.argmax(
        logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    tokens = torch.cat([tokens, next_token], dim=1)

    # ----------------------------
    # Decode
    # ----------------------------
    for _ in range(max_new_tokens - 1):
        logits, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            debug=debug,
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyTransformerLM(
        vocab_size=1000,
        dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        max_seq_len=128,
    ).to(device)

    input_ids = torch.tensor(
        [[10, 20, 30, 40]],
        device=device,
    )

    print("input_ids:")
    print(input_ids)

    print("\n========== naive ==========")

    output_naive = generate_naive(
        model,
        input_ids,
        max_new_tokens=5,
    )

    print(output_naive)

    print("\n========== kv cache ==========")

    output_cache = generate_kvcache(
        model,
        input_ids,
        max_new_tokens=5,
        debug=True,
    )

    print(output_cache)

    print("\noutputs equal:")
    print(torch.equal(output_naive, output_cache))


if __name__ == "__main__":
    main()

        
        
        