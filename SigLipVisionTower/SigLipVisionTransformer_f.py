import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ../Config/VisionConfig_f import VisionConfig


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: 'VisionConfig'):
        super().__init__()
        # ---------------------------------------------------------------------
        # Build 1x Transformer Block
        # ---------------------------------------------------------------------
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.dropout = config.attention_drop
        self.dmodel = config.dmodel
        #  64       = 768           //          12
        head_dim    = self.dmodel // self.num_attention_heads

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
        self.scale       = 1/math.sqrt(head_dim) 
        self.attn_dropout= nn.Dropout(config.attention_drop) # Apply dropout to attention weights
        self.attn_mlp    = nn.Linear(config.dmodel, config.dmodel)
        
        # ---------------------------------------------------------------------
        # 3) Post self-Attention & Pre-MLP LayerNorm2 and MLP
        # ---------------------------------------------------------------------
        self.layer_norm2 = nn.LayerNorm(config.dmodel, eps=config.layer_norm_eps)
        self.mlp_fc1     = nn.Linear(config.dmodel, config.dff_inner_dim)
        self.mlp_fc2     = nn.Linear(config.dff_inner_dim, config.dmodel)

    def forward(self, transformer_input):
        
        # [batch_size, seq_len, dmodel] 
        residual = transformer_input
        x = self.layer_norm1(transformer_input)
        
        
        # Compute Q, K, V
        # [batch_size, seq_len, dmodel] 
        # -> [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        batch_size, seq_len, embed_dim = transformer_input.shape
        head_dim = embed_dim // self.num_attention_heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1,2)
        k = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1,2)
        v = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1,2)
        
        # Attention scores
        # [batch_size, num_heads, seq_len, head_dim] . [batch_size, num_heads, head_dim, seq_len]
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights =self.attn_dropout(attn_weights)
        
        # attention output
        # [batch_size, num_heads, seq_len, seq_len] . [batch_size, num_heads, seq_len, head_dim]
        # [batch_size, num_heads, seq_len, head_dim]
        x = torch.matmul(attn_weights, v)
        # reshape back to dmodel
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, embed_dim]
        x = x.transpose(2, 1).reshape(batch_size, seq_len, embed_dim)
        # attn_output=attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        x = self.attn_mlp(x)
        x = residual + x
        
        # MLP block
        residual = x
        attn_output = self.layer_norm2(x)
        attn_output = self.mlp_fc1(attn_output)
        attn_output = F.gelu(attn_output, approximate = "tanh")
        attn_output = self.mlp_fc2(attn_output)
        attn_output = attn_output + residual
        
        #[batch_size, seq_len, embed_dim]
        return attn_output, attn_weights