from naive.simple_transformer import TinyTransformerLM

class EagleTargetWrapper(TinyTransformerLM):
    def forward(self, input_ids, return_hidden=False):
        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden

        return logits