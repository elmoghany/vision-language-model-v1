import torch

class ImageTextFusion:
    """
    This class replicates the logic of the original
    _merge_image_features_with_input_ids method, but now
    in a separate file and wrapped into a callable class.
    """

    def __init__(self, config):
        """
        config: typically a FusionConfig or a structure
                containing at least:
                  - config.hidden_size
                  - config.image_token_index
                  - config.pad_token_id
        """
        self.config = config
    
    def __call__(
        self,
        image_features: torch.Tensor,  # [batch_size, image_tokens, embed_dim]
        input_embeds: torch.Tensor,    # [batch_size, seq_length, embed_dim]
        input_ids: torch.Tensor,       # [batch_size, seq_length]
        attn_mask: torch.Tensor,       # [batch_size, seq_length]
        kv_cache=None
    ):
        """
        Merges scaled image_features into input_embeds where
        input_ids == <image_token>. Zero out padding as needed,
        and build a minimal causal mask + position_ids.
        
        Returns:
          final_embedding, updated_attn_mask, position_ids
        """
        batch_size, seq_length = input_ids.shape
        _, _, embed_dim        = image_features.shape

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


        ##############################
        # Build a minimal causal mask
        ##############################
        query_len = input_embeds.shape[-2]  # typically seq_length
        # If no kv_cache or it's empty, we do a prefill scenario
        if kv_cache is None or kv_cache.num_items() == 0:
            # shape => [batch_size, query_len, query_len]
            causal_mask = torch.zeros(
                (batch_size, query_len, query_len),
                dtype=final_embedding.dtype,
                device=final_embedding.device
            )
        else:
            # Generating tokens one by one => single step
            assert query_len == 1, "For incremental gen, expect single query token."
            kv_len = kv_cache.num_items() + query_len
            # shape => [batch_size, query_len, kv_len]
            causal_mask = torch.zeros(
                (batch_size, query_len, kv_len),
                dtype=final_embedding.dtype,
                device=final_embedding.device
            )

        # Expand for heads => [B, 1, Q, K]
        causal_mask = causal_mask.unsqueeze(1)

        ########################
        # Build position_ids
        ########################
        if kv_cache is not None and kv_cache.num_items() > 0:
            # Single-step: position is the last valid token
            position_ids = attn_mask.sum(dim=-1)
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # For each token, cumsum the mask
            # if mask == 0 => position = 1
            position_ids = attn_mask.cumsum(-1).masked_fill_((attn_mask == 0), 1)

        # Return merged embeddings, updated mask, position_ids
        return final_embedding, causal_mask, position_ids
