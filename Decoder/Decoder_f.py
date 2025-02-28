from DecoderLayer_f import DecoderLayer
from ..Config.TextConfig_f import TextConfig
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config: 'TextConfig', layer_idx):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx=i) for i in range(config.num_layers)]
        )
    
    def forward(self, x, attn_mask, pos_ids, kv_cache):
        # x: [B, T, Hidden_Size]
        for layer in self.layers:
            x = layer(x, attn_mask, pos_ids, kv_cache)
        return x