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
        self.scale       = math.sqrt(1/config.head_dim) 
        self.dropout     = nn.Dropout(config.attention_drop) # Apply dropout to attention weights
        self.attn_mlp    = nn.Linear(config.dmodel, config.dmodel)
        
        # ---------------------------------------------------------------------
        # 3) Post self-Attention & Pre-MLP LayerNorm2 and MLP
        # ---------------------------------------------------------------------
        self.layer_norm2 = nn.LayerNorm(config.dmodel, eps=config.layer_norm_eps)
        self.mlp_fc1     = nn.Linear(config.dmodel, config.dff_inner_dim)
        self.mlp_fc2     = nn.Linear(config.dff_inner_dim, config.dmodel)

    def forward(self, transformer_input):
        
        residual = transformer_input
        x = self.layer_norm1(transformer_input)
        
        
        # Compute Q, K, V
        # [batch_size, seq_len, dmodel] 
        # -> [batch_size, seq_len, num_heads, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        batch_size, seq_len, embed_dim = transformer_input
        head_dim = embed_dim // self.num_attention_heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1,2)
        k = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1,2)
        v = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, head_dim).transpose(1,2)
        
        # Attention scores
        # [batch_size, num_heads, seq_len, head_dim] . [batch_size, num_heads, head_dim, seq_len]
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # attention output
        # [batch_size, num_heads, seq_len, seq_len] . [batch_size, num_heads, seq_len, head_dim]
        # [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        # reshape back to dmodel
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(2, 1).reshape(batch_size, seq_len, embed_dim)
        # attn_output=attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.attn_mlp(attn_output)
        x = residual + attn_output
        
        # MLP block
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp_fc1(x)
        x = F.gelu(x, approximate = "tanh")
        x = self.mlp_fc2(x)
        x = x + residual
        
        #[batch_size, seq_len, embed_dim]
        return x