from KVCacheAttention_f import KVCacheAttention
from ../Config/TextConfig_f import TextConfig
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config: 'TextConfig', layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = KVCacheAttention(config=config, layer_idx=layer_idx)
        
        #                    Input (X)
        #               │
        #         ┌─────┴─────┐
        #         │           │
        #         │           │
        #         ▼           ▼
        #   Up=f(Wu⋅X)    Gate=f(Wg.X)
        #   nn.Linear      nn.Linear
        #  (Up Projection) (Gate Projection)
        #         │           │
        #      Activation    Sigmoid (σ)
        #  (e.g., GeLU/f(x))   │
        #         │           │
        #         └─────┬─────┘
        #               │  Element-wise
        #               │  multiplication
        #               ▼
        #        Gated Representation
        #               │
        #         nn.Linear (Down Projection)
        #               │
        #            Output (Y)

        # MLP
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        
        # RMS LayerNorm
        self.eps = config.rms_norm_eps
        self.weight_pre = nn.Parameter(torch.zeros(self.hidden_size))
        self.weight_post = nn.Parameter(torch.zeros(self.hidden_size))
    
    def forward(self, x, ):
        pass