import torch
import torch.nn as nn
import torch.nn.functional as F
from ../Config/VisionConfig_f import VisionConfig
from SigLipVisionTransformer_f import SigLipVisionTransformer

# 1. Create a patch of embeddings
# 2. Add positional embedding
# 3. Apply self-attention
# 4. Multi-head Attention
# 5. MLP
# 6. Output shape [batch_size, num_patches, dmodel]

class SigLipVisionModel(nn.Module):
    def __init__(self, config: 'VisionConfig'):
        super().__init__()
        self.config     = config
        
        # ---------------------------------------------------------------------
        # 1) Patch Embedding
        # ---------------------------------------------------------------------
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,  # 3
            out_channels= config.dmodel,        # 768
            kernel_size = config.patch_size,    # 16
            stride      = config.patch_size,    # 16
            padding     = 0
        )

        # e.g., (224//16) ^ 2 = 14*14 = 196 patches 
        num_patches = (config.image_size // config.patch_size) ** 2

        # ---------------------------------------------------------------------
        # 2) Positional Embedding
        # ---------------------------------------------------------------------
        # model learns an embedding for each position during training 
        # rather than computing fixed sin/cos values.
        self.positional_embedding = nn.Embedding(
            num_embeddings  = num_patches,  # Number of patches     # 196
            embedding_dim   = config.dmodel,# Embedding dimension   # 768
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
        # 3) Build Nx Transformer Blocks
        # ---------------------------------------------------------------------
        # 12 layers of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SigLipVisionTransformer(config) for _ in range(config.num_layers)
        ])

        # ---------------------------------------------------------------------
        # 4) Final Layer Norm 
        # ---------------------------------------------------------------------
        # Final layer norm after all transformer blocks
        self.final_layer_norm = nn.LayerNorm(config.dmodel, eps=config.layer_norm_eps)


    # input pixel_values
    # [batch_size, num_channels, height, width]    => [batch_size, num_patches, dmodel]
    # [batch_size, 3, 224, 224]                             => [batch_size, 196, 768]
    def forward(self, pixel_values):
        # ---------------------------------------------------------------------
        # 1) Patch Embedding
        # ---------------------------------------------------------------------
        # patch embedding = conv2d
        # [batch_size, num_channels, height, width]   
        # [batch_size, 3, 224, 224]                             
        patch_embeds = self.patch_embedding(pixel_values)
        
        # [batch_size, num_channels, height, width]   
        # [batch_size, 3, 224, 224]                             
        # conv2d_size = Lower((img_size + 2P - K)/S) + 1
        # lower(224 + 2*0 - 16 / 16) + 1 = 14
        # num_channels 3 => dmodel 768
        # [batch_size, 768, 14, 14]
        
        # flatten the image size
        # [batch_size, num_channels=dmodel, num_patches_flattened]
        # [batch_size, 768, 196]
        x = patch_embeds.flatten(2,-1)
        
        # transpose to [batch_size, num_patches_flattened, dmodel]
        x = x.transpose(1,2)
        
        # ---------------------------------------------------------------------
        # 2) Positional Embedding
        # ---------------------------------------------------------------------
        # add positional embedding to each patch embedding
        x = x + self.positional_embedding(self.positional_embedding_buffer)
        
        # ---------------------------------------------------------------------
        # 3) Build Nx Transformer Blocks
        # ---------------------------------------------------------------------
        for transformer_block in self.transformer_blocks:
            x,_ = transformer_block(x)
        
        # ---------------------------------------------------------------------
        # 4) Final Layer Norm 
        # ---------------------------------------------------------------------
        x = self.final_layer_norm(x)
        
        # [batch_size, num_patches_flattened, dmodel]
        return x