import torch

class KVCache:
    def __init__(self):
        # List of tuples (key, value) per layer.
        self.cache = []
        
    def num_items(self):
        if not self.cache:
            return 0
        #  0    1         -2         -1
        # [B, num_heads, seq_len, head_dim]
        #           [key][value][seq_len = # of tokens]
        return self.cache[0][0].shape[-2]
    
    def update(self, key, value, layer_idx):
        while len(self.cache) <= layer_idx:
            self.cache.append((None,None))
        
        old_key, old_value = self.cache[layer_idx]
        if old_key is None:
            self.cache[layer_idx] = (key, value)
        else:
            # If keys already exist, append the new keys/values along the sequence dimension.
            # The sequence dimension is at index -2.
            self.cache[layer_idx] = (torch.cat([old_key, key], dim=-2), torch.cat([old_value, value], dim=-2))
        return self.cache[layer_idx]