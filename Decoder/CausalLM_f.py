import torch
import torch.nn as nn
from ..Config.TextConfig_f import TextConfig
from Decoder_f import Decoder

class CausalLM(nn.Module):
    def __init__(self, config: 'TextConfig'):
        super().__init__()
        self.decoder = Decoder(config)
        self.padding_idx = config.pad_token_id


        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx
        )
        
        # tie weights
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, attn_mask, pos_ids, input_embeds, kv_cache):
        
        x = self.decoder(attn_mask, pos_ids, input_embeds, kv_cache)
        
        # post decoder rms normalization
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * (1.0 + self.weight_post)

        # lm head layer - post decoder
        logits = self.lm_head(x).float()
        return_data =  {"logits": logits}
        
        if kv_cache is not None:
            # return the updated cache
            return_data["kv_cache"] = kv_cache
            
        return return_data