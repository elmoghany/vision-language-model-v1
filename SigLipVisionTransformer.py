import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import SigLipVisionConfig


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        # ---------------------------------------------------------------------
        # Build 1x Transformer Block
        # ---------------------------------------------------------------------
        self.config = config
        #  64       = 768           //          12
        head_dim    = config.dmodel // config.num_attention_heads

        # ---------------------------------------------------------------------
        # 1) Pre-Attention LayerNorm and Q/K/V projections
        # ---------------------------------------------------------------------
        self.layer_norm1 = nn.LayerNorm(config.dmodel, eps=config.layer_norm_eps)
        self.q_proj      = nn.Linear(config.dmodel, config.dmodel)
        self.k_proj      = nn.Linear(config.dmodel, config.dmodel)
        self.v_proj      = nn.Linear(config.dmodel, config.dmodel)

        # ---------------------------------------------------------------------
        # 2) self-Attention 
        # ---------------------------------------------------------------------
        self.scale       = math.sqrt(1/config.head_dim) 
        self.dropout     = nn.Dropout(config.attention_drop) # Apply dropout to attention weights
        self.attn_out    = nn.Linear(config.dmodel, config.dmodel)
        
        # ---------------------------------------------------------------------
        # 3) Post self-Attention & Pre-MLP LayerNorm2 and MLP
        # ---------------------------------------------------------------------
        self.layer_norm2 = nn.LayerNorm(config.dmodel, eps=config.layer_norm_eps)
        self.mlp_fc1     = nn.Linear(config.dmodel, config.dff_inner_dim)
        self.mlp_fc2     = nn.Linear(config.dff_inner_dim, config.dmodel)

    def forward(self, transformer_input):
        
        batch_size, seq_len, embed_dim = transformer_input
        head_dim 