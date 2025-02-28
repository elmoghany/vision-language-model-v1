from ..Config.TextConfig_f import TextConfig
from RotaryEmbedding_f import RotaryEmbedding, apply_rotary
from KVCache_f import KVCache
import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat_kv(hidden_states, n_rep: int):
    if n_rep == 1:
        return hidden_states
    
    # num_heads is the number of KV heads.
    batch_size, num_heads, seq_len, head_dim = hidden_states.shape

    # Add a new dimension at position 2 (to hold repetitions).
    # New shape: [B, num_kv_heads, 1, seq_len, head_dim]
    hidden_states = hidden_states.unsqueeze(2)
    
    # hold the value of repitition to be used later
    hidden_states = hidden_states.expand(batch_size, num_heads, n_rep, seq_len, head_dim)

    # unfold the repititions by multiplying by num_of_kv_heads
    return hidden_states.reshape(batch_size, num_heads * n_rep, seq_len, head_dim)

class KVCacheAttention(nn.Module):
    def __init__(self, config: 'TextConfig', layer_idx = None):
        super().__init__()
        
        # config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # query
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.attention_dropout
        self.scale = 1 / self.head_dim.sqrt()
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        # projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # rotary embedding
        self.rotary = RotaryEmbedding(self.head_dim, self.max_position_embeddings, self.rope_theta, "cuda")
        
    def forward(self, x, attn_mask, pos_ids, kv_cache: 'KVCache' = None, layer_idx=0):
        
        ######################################
        #####--- (1) Q, K, V Projection ---#####
        batch_size, seq_len, _ = x.size()
        
        # Input x: [B, seq_len, hidden_size]
        # Input x: [B, seq_len, num_heads, head_dim]
        # Input x: [B, num_heads, seq_len, head_dim]
        
        # Query projection
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Key and Value Projection: Project x into key and value representations.
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        ######################################
        #####--- (2) Rotary Embeddings ---#####
        # [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary(q, pos_ids, seq_len)
        # Shape becomes: [B, 1, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        # Rotated q: [B, num_heads, seq_len, head_dim]
        # Rotated k: [B, num_kv_heads, seq_len, head_dim] 
        q, k = apply_rotary(q, k, cos, sin)
        
        ######################################
        #####--- (3)  KV Cache Update ---#####
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_idx)
            
        #####--- (4)  Repeat Keys and Values for Multi-Query Attention ---#####
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        
        ######################################
        #####--- (5)  Scaled Dot-Product Attention ---#####
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores + attn_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        # [B, num_heads, seq_len, head_dim]
        attn_out = torch.matmul(attn_weights, v)
        
        
        ######################################
        #####--- (6)  Final Projection ---#####
        # [B, num_heads, seq_len, head_dim]
        attn_out = attn_out.transpose(1, 2)
        # [B, seq_len, num_heads, head_dim]
        attn_out = attn_out.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        # [B, seq_len, num_heads * head_dim]
        attn_out = self.o_proj(attn_out)
        
        return att_out, attn_weights