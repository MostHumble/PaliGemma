from typing import Any, List, Union, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from siglip import  SiglipVisionConfig, SiglipVisionModel

class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

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
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
         
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # bs, num_patches, embed_dim -> batch_Size, num_patches, projection_dim
        hidden_states = self.linear(image_features)
        return hidden_states
    
class PaliGemmaForConditonnalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = SiglipVisionModel(config.vision_config)
        self.linear = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor, 
        input_embeds: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        kv_cache: Optional[KVCache] = None
    ):
        embed_dim = image_features.size(-1)
        bs, seq_len = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device
        # bs, seq_len, hidden_dim
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # placeholder for the embeddings
        final_embedding = torch.cat([scaled_image_features, input_embeds[:,scaled_image_features.size(1):]], dim=1)
        
        # masking

        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]
    
        # prefill phase
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full(
                (bs, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )

        # generation phase
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
    
            causal_mask = torch.full(
                (bs, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # bs, q_len, KV_len -> bs, num_heads_q, q_len, KV_len
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None

    ) -> tuple :  
        
        # bs, seq_len (num_patches+prompt seq lenght + 2 (bos, \n)), hidden_dim
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # bs, c, h, w -> bs, num_patches, vision_embed_dim
        image_embeds = self.vision_encoder(pixel_values.to(input_embeds.dtype))

        # bs, seq_len, vision_embed_dim - > bs, seq, hidden_dim
        image_embeds = self.linear(image_embeds)

        # merge the image embeds and text embeds
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_embeds, input_embeds, input_ids, attention_mask, kv_cache)

        # feed the whole thing to a LM

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache
        )

        return outputs