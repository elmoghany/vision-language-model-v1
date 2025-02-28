from SigLipVisionConfig_f import SigLipVisionConfig
from TextConfig_f import TextConfig

class FusionConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        device = "cuda",
        **kwargs,
    ):
        super().__init__()
        
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.device = device
        self.pad_token_id = pad_token_id

        self.vision_config = vision_config
        self.vision_config = SigLipVisionConfig(**vision_config)
        self.vision_config.projection_dim = projection_dim

        self.text_config = text_config
        self.text_config = TextConfig(**text_config, pad_token_id=pad_token_id)
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        
        self.vocab_size = vocab_size
        self.vocab_size = self.text_config.vocab_size

