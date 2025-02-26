import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self,
                head_dim: int = 256, 
                max_position_embeddings: int = 2048, 
                base = 10000, 
                device=None):
        super().__init__()
        
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device
        
        # Calculate the theta according to the formula 
        # theta_i = base^(-2i/dim) where i = 0, 1, 2, ..., dim // 2
        # theta_i = base^(-i/dim) where i = 0, 2, 4, ..., dim // 2
        inv_freq = 1/ (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq_buffer", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len: int):
        
        self.inv_freq_buffer.to(x.device)
        # x: [batch_size, num_attention_heads, seq_len, head_size]
        # position_ids: [batch_size, seq_len]

        # position_ids.shape[0] = batch_size
        # -1 => remains unchanged
        # 1 => last dim to be 1
        # Expand inv_freq to shape [batch_size, head_dim//2, 1]
        inv_freq_expanded = self.inv_freq_buffer[None, :, None].expand(position_ids.shape[0], -1, 1)

        # Create a tensor of positions: 
        # [0, 1, ..., seq_len-1]
        # unsequeeze(1) => Shape: [seq_len, 1]
        # position_ids_expanded = torch.arange(seq_len, device=self.device).unsqueeze(1).float()
        # ================================
        # Expand position_ids to shape [batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        with torch.autocast(device_type=self.device, enabled=False):

            # multiply positions by the inverse frequencies
            # inv_freq [batch_size, head_dim//2, 1] @ position_ids_expanded [batch_size, 1, seq_len]
            # shape: [seq_len, dim/2]
            # ================================
            # [batch_size, head_dim//2, seq_len]
            pos_inv_freqs = inv_freq_expanded @ position_ids_expanded
            
            # [batch_size, seq_len, head_dim//2]
            pos_inv_freqs = pos_inv_freqs.transpose(1, 2)
            
            # Duplicate the frequency values to form a full vector of length dim
            # shape: [seq_len, "dim/2"] + [seq_len, "dim/2"]  dim=1
            # shape = [seq_len, "dim"]
            # =============================
            # Resulting shape: [batch_size, seq_len, head_dim]
            full_dim = torch.cat((pos_inv_freqs,pos_inv_freqs), dim=1)
            
            # Return cosine and sine values
            cos = full_dim.cos()
            sin = full_dim.sin()
        
        return cos, sin