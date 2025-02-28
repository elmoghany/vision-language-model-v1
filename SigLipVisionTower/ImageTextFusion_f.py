import torch
import torch.nn as nn

###############################################################################
# 2. Prepend concatenation of text and image tokens
###############################################################################
class ImageTextFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        # text_embeds: [B, T, D], image_embeds: [B, I, D]
        # Prepend image embeddings to the text embeddings: [B, I+T, D]
        return torch.cat([image_embeds, text_embeds], dim=1)