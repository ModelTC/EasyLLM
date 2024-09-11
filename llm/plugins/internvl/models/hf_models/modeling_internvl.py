# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
from .modeling_internlm2 import InternLM2ForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_eva_vit import EvaCLIPVisionModel
# from ...datas.conversation import get_conv_template
from llm.models.hf_models.sequence import (get_sequence_parallel_world_size,
                                           get_sequence_parallel_rank,
                                           get_sequence_parallel_group)


logger = logging.get_logger(__name__)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size, assuming square window

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, 'H and W must be divisible by window_size'

    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H * W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionEncoderLayer', 'LlamaDecoderLayer', 'LlamaForCausalLM']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.image_fold = config.image_fold
        self.ps_version = config.ps_version

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if config.vision_config.architectures[0] == 'InternVisionModel':
                self.vision_model = InternVisionModel(config.vision_config)
            elif config.vision_config.architectures[0] == 'EvaCLIPVisionModel':
                self.vision_model = EvaCLIPVisionModel(config.vision_config)
            else:
                raise NotImplementedError(f'{config.vision_config.architectures[0]} is not implemented.')
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        vit_hidden_size_scale = 1 / (self.downsample_ratio * self.downsample_ratio)

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(int(vit_hidden_size * vit_hidden_size_scale)),
            nn.Linear(int(vit_hidden_size * vit_hidden_size_scale), llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        if config.force_image_size != config.vision_config.image_size:
            self.vision_model.resize_pos_embeddings(
                old_size=config.vision_config.image_size,
                new_size=config.force_image_size,
                patch_size=config.vision_config.patch_size
            )

        self.img_context_token_id = None

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        self.SP_SIZE = get_sequence_parallel_world_size()
        self.SP_RANK = get_sequence_parallel_rank()
        self.SP_GROUP = get_sequence_parallel_group()
        self._init_mlp()

    def _init_mlp(self):
        self.mlp1[0].reset_parameters()
        self.mlp1[1].reset_parameters()
        self.mlp1[3].reset_parameters()

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def split_for_sp(self, tensor, size, dim, rank):
        tensor_list = torch.split(tensor, size // self.SP_SIZE, dim=dim)
        return tensor_list[rank].contiguous()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cu_seqlens=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        image_flags = image_flags.view(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # if image_flags.sum() != 0:
        vit_embeds = self.extract_feature(pixel_values)

        if self.SP_SIZE > 1 and len(image_flags) != pixel_values.shape[0]:
            vit_embeds_list = [torch.zeros_like(vit_embeds) for _ in range(self.SP_SIZE)]
            dist.all_gather(vit_embeds_list, vit_embeds, group=self.SP_GROUP)
            vit_embeds = torch.cat(vit_embeds_list, dim=0)

        vit_embeds = vit_embeds[image_flags == 1]
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        if self.SP_SIZE > 1 and len(image_flags) != pixel_values.shape[0]:
            length = input_embeds.shape[0]
            input_embeds = self.split_for_sp(input_embeds, length, 0, self.SP_RANK)
            labels = self.split_for_sp(labels, length, 1, self.SP_RANK)
            position_ids = self.split_for_sp(position_ids, length, 1, self.SP_RANK)
            N = N // self.SP_SIZE

        input_embeds = input_embeds.reshape(B, N, C)
        # else:
        #     vit_embeds = self.extract_feature(pixel_values, no_img=True)
        #     temp = input_embeds.clone()
        #     temp += vit_embeds
        #     input_embeds = temp
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # cu_seqlens = cu_seqlens.view(-1)
            # long_data = (cu_seqlens[1] - cu_seqlens[0]) > 30000
            # if get_sequence_parallel_world_size() > 1 and long_data:
            #     sp_group = get_sequence_parallel_group()
            #     num_tokens = (shift_labels != -100).sum()
            #     loss = reduce_sequence_parallel_loss(loss, num_tokens, sp_group)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.image_fold:
            image_size = pixel_values.size(-1)  # B, C, H, W
            pixel_values = window_partition(pixel_values, window_size=image_size // self.image_fold)  # 4B, C, H/2, W/2

        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.image_fold:
            vit_embeds = window_reverse(vit_embeds, window_size=image_size // (self.image_fold * self.patch_size),
                                        H=image_size // self.patch_size, W=image_size // self.patch_size)

        # if torch.distributed.get_rank() == 0:
        #     print("before pixel shuffle:", vit_embeds.shape)
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # if torch.distributed.get_rank() == 0:
        #     print("after pixel shuffle:", vit_embeds.shape)
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def chat(self, tokenizer, pixel_values, question, generation_config,
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        if tokenizer.convert_tokens_to_ids('<|im_end|>') != 0:
            eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')  # 92542, InternLM2
        else:
            eos_token_id = tokenizer.eos_token_id

        # template = get_conv_template(self.template)
        template = None
        image_bs = pixel_values.shape[0]
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * image_bs + IMG_END_TOKEN
        print(f'dynamic ViT batch size: {image_bs}')
        template.append_message(template.roles[0], image_tokens + '\n' + question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split('<|im_end|>')[0].strip()  # for InternLM2
        query_to_print = query.replace(image_tokens, '<image>')
        print(query_to_print, response)
        return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
