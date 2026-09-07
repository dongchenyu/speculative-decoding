为什么用了 KV Cache 后 RoPE 还需要特殊处理？

KV Cache 保存的是历史 token 已经计算好的 K/V，其中 K 通常已经根据该 token 的绝对 position 做过 RoPE。因此 decode 时历史 K 不需要重新旋转，只需要给当前新 token 的 Q/K 使用其在完整序列中的 position 做 RoPE，然后把新的 rotated K append 到 K cache。由于 decode 输入通常只有一个 token，不能直接用 arange(current_seq_len) 得到 position，而需要根据 past_len 计算真实位置，比如 position = past_len。

Prefill 阶段一次计算所有 prompt token，并把每一层已经做过 RoPE 的 K 和普通 V 存入 KV Cache；decode 阶段每次只处理新 token，根据 cache 长度得到它的全局 position，对新 Q/K 做 RoPE，再把新 K/V append 到对应层的 cache，然后用新 Q 对完整历史 K/V 做 attention。