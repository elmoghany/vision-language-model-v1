from FusionConfig_f import FusionConfig
import torch
import torch.nn as nn

###############################################################################
# 1. Linear projection for the image tokens
###############################################################################
class ImageProjection(nn.Module):
    def __init__(self, config: 'FusionConfig'):
        super().__init__()
        # Projects image tokens (of dimension image_token_dim)
        # into the text hidden space (projection_dim, which should equal text_hidden_size)
        # 768 x 2048
        self.linear = nn.Linear(config.vision_config.dmodel, config.vision_config.projection_dim, bias=True)

    # image tokens == image features
    # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
    def forward(self, image_tokens):
        return self.linear(image_tokens)

