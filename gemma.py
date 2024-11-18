from typing import Any, List, Union, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from siglip import  SiglipVisionConfig, SiglipVisionModel



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