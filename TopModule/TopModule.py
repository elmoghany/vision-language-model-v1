import torch
import torch.nn as nn
from PIL import Image

### Import Config ###
from ..Config.VisionConfig_f import VisionConfig
from ..Config.TextConfig_f import TextConfig
from ..Config.FusionConfig_f import FusionConfig 

### Import SigLip Vision Tower ###
from ..SigLipVisionTower.ImageProcessor_f import ImageProcessor
from ..SigLipVisionTower.SigLipVisionModel_f import SigLipVisionModel
from ..SigLipVisionTower.ImageProjection_f import ImageProjection
from ..SigLipVisionTower.ImageTextFusion_f import ImageTextFusion
### Import Decoder ###
from ..Decoder.CausalLM_f import CausalLM

class TopModule(nn.Module):
    def __init__(self, config: 'FusionConfig'):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
        # 1) Image Processor (for resizing, normalization, etc.)
        self.image_processor = ImageProcessor(
            tokenizer=None,  # or pass any needed tokenizer references
            num_image_tokens=config.vision_config.projection_dim,  # or however you want to param
            image_size=config.vision_config.image_size
        )

        # 2) Vision tower
        self.vision_tower = SigLipVisionModel(config.vision_config)
    
        # 3) Image projection (Dimension alignment)
        self.vision_projector = ImageProjection(self.config)
        
        # 4) Language Model
        self.language_model = CausalLM(config.text_config)
        
        # 5) merging
        self.merger = ImageTextFusion(config)

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def forward(self, input_ids, images, attn_mask, kv_cache):
        

        # Step 0: Pre-process images if needed (using self.image_processor).
        if images is None:
            raise ValueError("No image provided")
        
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            # Process the list of PIL images.
            processed_dict = self.image_processor(
                text=["dummy text"], 
                images=images
            )
            pixel_values = processed_dict["pixel_values"]
        else:
            # Assume images are already in [B, 3, H, W] format.
            pixel_values = images

        # make sure input is right-padded
        assert torch.all(attn_mask == 1), "the input can not be padded"
        
        # 1) Convert text & image tokens -> embeddings
        # input ids     shape: (Batch_Size, Seq_Len)
        # input embeds  shape: (Batch_Size, Seq_Len, Hidden_Size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 2) Vision tower: pixel_values => [B, #patches, dmodel]
        # [Batch_Size, Channels, Height, Width]
        # [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values)
        
        # 3) Project => align dims with text hidden size
        # [Batch_Size, Num_Patches, Embed_Dim]
        # [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.vision_projector(selected_image_feature)
        
        # 4) Merge embeddings of the text & image tokens
        final_embeds, causal_mask, position_ids = self.merger(
            image_features = image_features, 
            input_embeds = input_embeds, 
            input_ids = input_ids, 
            attn_mask = attn_mask, 
            kv_cache = kv_cache
            )
        
        # 5) Language model forward
        x = self.language_model(
            attn_mask = causal_mask, 
            pos_ids = position_ids, 
            input_embeds = final_embeds, 
            kv_cache = kv_cache
            )
        
        return x