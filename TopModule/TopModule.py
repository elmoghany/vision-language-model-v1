import torch
import torch.nn as nn

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
        
        self.vision_tower = SigLipVisionModel(config.vision_config)
        self.vision_projector = ImageProjection(self.config)
        
        self.language_model = CausalLM(config.text_config)
        
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_image_features_with_input_ids(
            self, 
            image_features,     # [batch_size, image_tokens, embed_dim]
            input_embeds,       # [batch_size, seq_length,   embed_dim]
            input_ids,          # [batch_size, seq_length]
            attn_mask,          # [batch_size, seq_length]
            kv_cache
        ):
        _, _, embed_dim = image_features.shape
        batch_size, seq_length = input_ids.shape
        
        # so that image embeddings have a similar magnitude as text embeddings
        # [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)
        
        ########################
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        ########################
        final_embedding = torch.zeros(batch_size, seq_length, embed_dim)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        # True where the token is not <image_token> and also not pad_token_id. 
        # meaning = these positions correspond to normal text.
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        # True where the token is <image_token>.
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        # True where the token is a padding token 
        # (meaning it’s not actual content).
        pad_mask = input_ids == self.pad_token_id
        # example
        # input_ids for one batch element is [10, 256000, 87, 256000, 0]
        # config.image_token_index = 256000
        # pad_token_id = 0
        # text_mask = [True, False, True, False, False]
        # image_mask = [False, True, False, True, False]
        # pad_mask = [False, False, False, False, True]

        # We need to expand the masks to the embedding dimension 
        # [batch_size, seq_length]
        # [batch_size, seq_length, 1]
        # [batch_size, seq_length, embed_dim]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # Insert image embeddings. 
        # We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ###################################
        #### CREATE THE ATTENTION MASK ####
        ###################################
        query_len = input_embeds.shape[-1]
        
        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token
            # pre-token phase
            # 0 in an attention mask typically means “no masking
            # This only works when we have no padding
            causal_mask = torch.zeros((batch_size, query_len, query_len))#, dtype=dtype, device=device)
        else:
            # Since we are generating tokens, the query must be one single token
            assert query_len == 1
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            kv_len = kv_cache.num_items() + query_len
            causal_mask = torch.zeros((batch_size, query_len, kv_len))#, dtype=dtype, device=device)
            
        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attn_mask.sum(dim=-1)
            if position_ids.dim() == 1:
                positions_ids = positions_ids.unsqueeze(0)
        else:
        # Create a position_ids based on the size of the attention_mask
        # For masked tokens, use the number 1 as position.
            position_ids = (attn_mask.cumsum(-1)).masked_fill_((attn_mask == 0), 1)

        return final_embedding, causal_mask, position_ids

    def forward(self, input_ids, pixel_values, attn_mask, kv_cache):
        # make sure input is right-padded
        assert torch.all(attn_mask == 1), "the input can not be padded"
        
        # 1. Input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 2. Merge Text & Images
        # [Batch_Size, Channels, Height, Width]
        # [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values)
        # [Batch_Size, Num_Patches, Embed_Dim]
        # [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.vision_projector(selected_image_feature)
        
        # Merge the embeddings of the text & image tokens
        input_embeds, attn_mask, position_ids = self._merge_image_features_with_input_ids(image_features, input_embeds, input_ids, attn_mask, kv_cache)
        
        x = self.language_model(attn_mask, position_ids, input_embeds, kv_cache)
        
        return x