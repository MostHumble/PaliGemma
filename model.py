from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self, 
            hidden_size: int = 768,
            intermediate_size: int = 3072,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            num_channels: int = 3, # R,G,B
            image_size: int=224, 
            patch_size: int=16, # 16x16 patches: (ex: 224/16 = 14x14 patches)
            layer_norm_eps: float = 1e-6, 
            attention_dropout: float = 0.0,
            num_image_tokens: int = None, # number of image embeddings that are generated from the image
            **kwargs
        ) -> None:

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels, # RGB
            out_channels=config.hidden_size, 
            kernel_size=config.patch_size,  # 16x16 patches
            stride=config.patch_size, # non-overlapping patches
            padding='valid' # no padding
        )
        self.position_embeddings = nn.Embedding(self.num_patches, config.hidden_size)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False
            )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_channels, height, width) -> (batch_size, hidden_size, num_patches)
        x = self.patch_embeddings(images).flatten(2).transpose(1, 2)
        x = torch.cat([self.position_embeddings, x], dim=1)
        return x


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config

        self.embed = SiglipVisionEmbedding(config)
        self.encode = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_channels, height, width) -> (batch_size, num_patches, embed_dim)
        x = self.embed(images)
        x = self.encode(x)
        x = self.post_layernorm(x)
        return x

class SiglipVision(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self,  images: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_channels, height, width) -> (batch_size, num_patches, embed_dim)
        return self.vision_model(images=images)



        
        
