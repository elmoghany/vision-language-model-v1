import torch
import torch.nn as nn
import torch.nn.functional as F
import SigLipVisionConfig
import SigLipVisionTransformer

# 1. Create a patch of embeddings
# 2. Add positional embedding
# 3. Apply self-attention
# 4. Multi-head Attention
# 5. MLP
# 6. Output shape [batch_size, num_patches, dmodel]

class SigLipVisionModel(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config     = config
        
        # ---------------------------------------------------------------------
        # 1) Patch + Position Embedding
        # ---------------------------------------------------------------------
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels= config.dmodel,
            kernel_size = config.patch_size,
            stride      = config.patch_size,
            padding     = 0
        )

        num_patches = (config.image_size // config.patch_size) ** 2
        
        # model learns an embedding for each position during training 
        # rather than computing fixed sin/cos values.
        self.positional_embedding = nn.Embedding(
            num_embeddings  = num_patches,  # Number of patches
            embedding_dim   = config.dmodel,# Embedding dimension
        )
        
        # using .expand((1, -1)), you reshape this tensor to [1, num_patches]
        # instead of having 1D tensor of shape [num_patches]
        # The extra dimension (with size 1) allows the positional indices 
        # to be broadcast correctly across the batch.
        
        # buffer => won’t be included in the model’s state dict
        self.register_buffer(
            "positional_embedding_buffer",
            torch.arange(num_patches).expand((1, -1)),
            persistent=False
        )

        # ---------------------------------------------------------------------
        # 2) Build Nx Transformer Blocks
        # ---------------------------------------------------------------------




    def forward(self, ):
        # [Batch_Size, Channels, Height, Width] -> 
        # [Batch_Size, Num_Patches, Embed_Dim]
        pass