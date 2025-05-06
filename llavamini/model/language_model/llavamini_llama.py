#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Callable
from functools import partial
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.utils import add_start_docstrings_to_model_forward, logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING


from ..llavamini_arch import LlavaMiniMetaModel, LlavaMiniMetaForCausalLM, MyRecurrent
import time

logger = logging.get_logger(__name__)

class LlavaMiniConfig(LlamaConfig):
    model_type = "llava_mini_llama"
    
    # https://huggingface.co/tomg-group-umd/huginn-0125/blob/main/config.json
    norm_eps = 1e-6
    n_layers_in_recurrent_block = 4
    embed_scale = 72.6636084983398
    init_values_std = 0.008703882797784892
    mean_recurrence = 32
    mean_backprop_depth = 8
    
    recurrent_in_compression = False
    recurrent_in_prefusion = False
    recurrent_as_prefusion = False
    recurrent_with_tcond = False
    recurrent_in_llm = False
    recurrent_in_prefusion_residue = False
    recurrent_in_llm_residue = False
    recurrent_in_llm_range = 4
    recurrent_as_llm = False
    recurrent_start_idx = 28
    activation_checkpoint_impl = 'per-iteration'
    

    @property
    def checkpoint(self) -> Callable:
        """Run SAC at your own risk :<"""
        attn_ops = [
            torch.ops.aten._scaled_dot_product_efficient_attention.default,  # type: ignore
            torch.ops.aten._scaled_dot_product_flash_attention.default,  # type: ignore
        ]
        try:
            from flash_attn import flash_attn_func  # type: ignore

            attn_ops.append(flash_attn_func)
        except ImportError:
            pass
        ops_to_save = [
            torch.ops.aten.mm.default,  # type: ignore
            *attn_ops,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,  # type: ignore # from comms
        ]

        return partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")

class LlavaMiniLlamaModel(LlavaMiniMetaModel, LlamaModel):
    config_class = LlavaMiniConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaMiniLlamaModel, self).__init__(config)
        self.n_layer = len(self.layers)
        self.llm_recurrent = None
        self.config = config
        self.recurrent_in_llm = config.recurrent_in_llm
        self.recurrent_in_llm_residue = config.recurrent_in_llm_residue
        self.recurrent_in_llm_range = config.recurrent_in_llm_range
        self.recurrent_as_llm = config.recurrent_as_llm
        self.recurrent_start_idx = config.recurrent_start_idx
        # self.create_recurrent_in_llm()
        # print("recurrent_in_llm_residue", self.recurrent_in_llm_residue, self.recurrent_in_llm_range)
        # print(self.recurrent_in_llm)
        # if config.recurrent_in_llm:
        #     self.llm_recurrent = HuginnRecurrent(config)
            
    def create_recurrent_in_llm(self):
        if self.recurrent_in_llm or self.recurrent_in_llm_residue:
            self.llm_recurrent = HuginnRecurrent(self.config)
        if self.recurrent_as_llm:
            start_idx = self.recurrent_start_idx
            recurrent = MyRecurrent(self.config)
            
            for idx in range(self.config.n_layers_in_recurrent_block):
                layer_ = self.layers[idx+start_idx]
                recurrent.core_block[idx].self_attn.q_proj.weight.data = layer_.self_attn.q_proj.weight.data.clone()
                recurrent.core_block[idx].self_attn.k_proj.weight.data = layer_.self_attn.k_proj.weight.data.clone()
                recurrent.core_block[idx].self_attn.v_proj.weight.data = layer_.self_attn.v_proj.weight.data.clone()
                recurrent.core_block[idx].self_attn.o_proj.weight.data = layer_.self_attn.o_proj.weight.data.clone()
                recurrent.core_block[idx].mlp.gate_proj.weight.data = layer_.mlp.gate_proj.weight.data.clone()
                recurrent.core_block[idx].mlp.up_proj.weight.data = layer_.mlp.up_proj.weight.data.clone()
                recurrent.core_block[idx].mlp.down_proj.weight.data = layer_.mlp.down_proj.weight.data.clone()
                recurrent.core_block[idx].input_layernorm.weight.data = layer_.input_layernorm.weight.data.clone()
                recurrent.core_block[idx].post_attention_layernorm.weight.data = layer_.post_attention_layernorm.weight.data.clone()
            del self.layers[start_idx:start_idx+self.config.n_layers_in_recurrent_block]

            self.llm_recurrent = recurrent

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.recurrent_as_llm and i==self.recurrent_start_idx and self.llm_recurrent is not None:
                layer_outputs = self.llm_recurrent(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
                if use_cache and i>self.recurrent_start_idx:
                    decoder_layer.self_attn.layer_idx = i-1
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

            hidden_states = layer_outputs[0]
            
            if self.recurrent_in_llm and self.llm_recurrent is not None and i==(self.n_layer//2):
                # print("Recurrent in LLM working...")
                hidden_states = self.llm_recurrent(hidden_states)[0]

            if self.recurrent_in_llm_residue and self.llm_recurrent is not None and i == (self.n_layer//2 - self.recurrent_in_llm_range//2):
                # print("Recurrent Residule In")
                recurrent_out = self.llm_recurrent(hidden_states)[0]
            if self.recurrent_in_llm_residue and self.llm_recurrent is not None and i == (self.n_layer//2 + self.recurrent_in_llm_range//2):
                # print("Recurrent Residule Out")
                hidden_states += recurrent_out

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.recurrent_as_llm and self.recurrent_start_idx==len(self.layers) and self.llm_recurrent is not None:
            hidden_states = self.llm_recurrent(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlavaMiniLlamaForCausalLM(LlamaForCausalLM, LlavaMiniMetaForCausalLM):
    config_class = LlavaMiniConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaMiniLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_mini_llama", LlavaMiniConfig)
AutoModelForCausalLM.register(LlavaMiniConfig, LlavaMiniLlamaForCausalLM)
