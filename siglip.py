import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,  # R,G,B
        image_size: int = 224,
        patch_size: int = 16,  # 16x16 patches: (ex: 224/16 = 14x14 patches)
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: int = None,  # number of image embeddings that are generated from the image
        **kwargs,
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
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.make_patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,  # RGB
            out_channels=config.hidden_size,  # embed_dim
            kernel_size=config.patch_size,  # 16x16 patches
            stride=config.patch_size,  # non-overlapping patches
            padding="valid",  # no padding
        )
        self.positionnal_embeddings = nn.Embedding(self.num_patches, config.hidden_size)

        # not sure what's the point of this, must investigate
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).unsqueeze(0),  # 1, num_patches
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # image of shape: batch_size, num_channels, height, width
        embeddings = self.make_patch_embeddings(
            images
        )  # batch, embed_dim, num_patches_h, num_patches_w
        embeddings = embeddings.flatten(2)  # batch, embed_dim, num_patches
        embeddings = embeddings.transpose(1, 2)  # batch, num_patches, embed_dim
        embeddings = embeddings + self.positionnal_embeddings(
            self.position_ids
        )  # batch, num_patches, embed_dim


class MLP(nn.Module):
    def _init__(self, in_features, intermidate_features):
        super().__init__()
        self.Linear1 = nn.Linear(in_features, intermidate_features)
        self.Linear2 = nn.Linear(intermidate_features, in_features)

    def forward(self, x):
        # batch_size, num_patches, intermidate_features
        x = self.Linear1(x)
        x = nn.functional.gelu(x, approximate="tanh")


class SiglipVisionEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layerNorm1 = nn.LayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps
        )
        self.layerNorm2 = nn.LayerNorm(
            normalized_shape=config.hidden_size, eps=config.layer_norm_eps
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.mlp = MLP(
            in_features=config.hidden_size,
            intermidate_features=config.intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shapes: batch_size, num_patches, embed_dim
        residual = x
        # apply first layerNorm (pre-LN)
        x = self.layerNorm1(x)
        # use attention mechanism, and add residual stream
        x = self.attn(x) + residual
        residual = x
        # apply second layerNorm
        x = self.layerNorm2(x)
        # apply linear layer and add residual stream
        x = self.mlp(x) + residual
        return x


class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_patches, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return x


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed = SiglipVisionEmbedding(config)
        self.encode = SiglipVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_channels, height, width) -> (batch_size, num_patches, embed_dim)
        x = self.embed(images)
        x = self.encode(x)
        x = self.post_layernorm(x)
        return x


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_channels, height, width) -> (batch_size, num_patches, embed_dim)
        return self.vision_model(images=images)
