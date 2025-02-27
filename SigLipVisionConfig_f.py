class SigLipVisionConfig:
    def __init__(
        self,
        dmodel              = 768,
        dff_inner_dim       = 3072,
        num_layers          = 12,
        num_attention_heads = 12,
        num_channels        = 3,
        image_size          = 224,
        patch_size          = 16,
        layer_norm_eps      = 1e-6,
        attention_drop      = 0.0,
        num_image_tokens: int = None,
        **kwargs
        ):
        super().__init__()
        self.dmodel             = dmodel
        self.dff_inner_dim      = dff_inner_dim
        self.num_layers         = num_layers
        self.num_attention_heads= num_attention_heads
        self.num_channels       = num_channels
        self.image_size         = image_size
        self.patch_size         = patch_size
        self.layer_norm_eps     = layer_norm_eps
        self.attention_drop     = attention_drop
        self.num_image_tokens   = num_image_tokens