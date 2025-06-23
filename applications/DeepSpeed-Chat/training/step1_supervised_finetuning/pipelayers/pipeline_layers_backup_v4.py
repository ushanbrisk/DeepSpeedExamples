'''
this code is to split qwen2.5b-1.5-instruct to multiple classes, for parallel processing
'''
import torch
from accelerate.utils import is_peft_model
from deepspeed.runtime.pipe import TiedLayerSpec, LayerSpec
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from training.step1_supervised_finetuning.grpo import get_reward_funcs, enable_gradient_checkpointing,check_module_requires_grad,PipelineGRPOEngine

from transformers import Qwen2ForCausalLM, Qwen2Model
import torch.nn as nn
import os
from torch.utils.checkpoint import checkpoint
# from inference.huggingface.zero_inference.utils import hidden_bytes
from torch.nn import functional as F
from trl.trainer.utils import selective_log_softmax
'''
1st layer, input: input_ids,
output:hidden_states, position_ids, 
'''
class PreEmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        if is_peft_model(model):
            self.embed_tokens = model.base_model.model.model.embed_tokens
            self.rotary_emb = model.base_model.model.model.rotary_emb
        else:
            self.embed_tokens = model.model.embed_tokens
            self.rotary_emb = model.model.rotary_emb
        self.weight = self.embed_tokens.weight

        # self.weight = self.embed_tokens.weight

        # print(f"PreEmbeddingPipeLayer Init(), memory: {id(self.embed_tokens.weight)}, pid:{os.getpid()}")
        # check_module_requires_grad(self.embed_tokens)

    def forward(self,  ipt):
        # input_ids, labels = ipt
        prompt_completion_ids,attention_mask = ipt

        inputs_embeds = self.embed_tokens(prompt_completion_ids) #[bz, seq_len] -> [bz, seq_len, hidden_v]
        cache_position = torch.arange(
            0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)  #[1 ,  seq_len]
        # causal_mask = self._update_causal_mask(
        #             attention_mask, inputs_embeds, cache_position, past_key_values=None, output_attentions=False
        # )
        causal_mask = attention_mask
        hidden_states = inputs_embeds   #[bz, seq_len, hidden_v]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings
        requires_grad_idx = torch.tensor([3]).to(hidden_states.device)  #here 3 different from [3], 3 may cause error in communication
        # print(f"pid: {os.getpid()},  PreEmbedding forward() called")
        # return requires_grad_idx, cos, sin, hidden_states, causal_mask, torch.tensor([logits_to_keep]).to(hidden_states.device), prompt_completion_ids
        return hidden_states, causal_mask


class DecoderPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM, layer_idx:int):
        super().__init__()
        if is_peft_model(model):
            self.layer = model.base_model.model.model.layers[layer_idx]
            self.rotary_emb = model.base_model.model.model.rotary_emb
            self._gradient_checkpointing_func = model.base_model.model.model._gradient_checkpointing_func
        else:
            self.layer = model.model.layers[layer_idx]
            self.rotary_emb = model.model.rotary_emb
            self._gradient_checkpointing_func = model.model._gradient_checkpointing_func
        self.layer_idx = torch.tensor(layer_idx)
        # print(f"DecoderPipeLayer Init(), index {self.layer_idx}")
        # check_module_requires_grad(self.layer)


    def forward(self, ipt):
        # print(f"decoder layer: {self.layer_idx} been called")
        hidden_states, causal_mask = ipt

        #reconstruct rotary embedding
        cache_position = torch.arange(
            0, 0 + hidden_states.shape[1], device=hidden_states.device
        )
        position_ids = cache_position.unsqueeze(0)  # [1 ,  seq_len]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        #above for rotary embedding

        # layer_outputs = checkpoint(self.layer, hidden_states,position_ids = position_ids,
        #                            cache_position = cache_position,
        #                            position_embeddings = position_embeddings,
        #                            use_reentrant = False)
        layer_outputs = self._gradient_checkpointing_func(self.layer.__call__,
                                   hidden_states,
                                   causal_mask=causal_mask,
                                   position_ids = position_ids,
                                   cache_position = cache_position,
                                   position_embeddings = position_embeddings,
                                   use_reentrant = False,
                                   use_cache=False)

        # layer_outputs = self.layer(hidden_states,
        #                            position_ids = position_ids,
        #                            cache_position = cache_position,
        #                            position_embeddings = position_embeddings)

        hidden_states = layer_outputs[0]
        # print(f"pid: {os.getpid()},  DecoderLayer.{self.layer_idx} forward() called")
        return hidden_states, causal_mask

class NormPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        if is_peft_model(model):
            self.norm = model.base_model.model.model.norm
        else:
            self.norm = model.model.norm

    def forward(self, ipt):
        hidden_states, causal_mask = ipt
        # hidden_states = self.norm(hidden_states)
        #here maybe we should not use checkpoint?
        hidden_states = checkpoint(self.norm, hidden_states, use_reentrant = False)
        # print(f"pid: {os.getpid()},  NormLayer forward() called")
        return hidden_states, causal_mask
#
# class LMHeadPipeLayer(torch.nn.Module):
#     def __init__(self, model:Qwen2ForCausalLM):
#         super().__init__()
#         if is_peft_model(model):
#             self.embed_tokens = model.base_model.model.model.embed_tokens
#         else:
#             self.embed_tokens = model.model.embed_tokens
#         self.weight = self.embed_tokens.weight
#
#     def forward(self, ipt):
#         hidden_states, labels = ipt
#         #logits = self.lm_head(hidden_states)
#         logits = torch.nn.functional.linear(hidden_states, self.embed_tokens.weight)
#         # print(f"pid: {os.getpid()},  LMHead forward() called")
#         return logits, labels

class LossPipeLayer(torch.nn.Module):
    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()

    def forward(self, ipt):
       logits, labels = ipt
       # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
       # Upcast to float if we need to compute the loss to avoid potential precision issues
       logits = logits.float()
       labels = labels.to(logits.device)
       # Shift so that tokens < n predict n
       labels = nn.functional.pad(labels, (0, 1), value=-100)
       shift_labels = labels[..., 1:].contiguous()

       # Flatten the tokens
       vocab_size = logits.shape[-1]
       logits = logits.view(-1, vocab_size)
       shift_labels = shift_labels.view(-1)
       # Enable model parallelism
       shift_labels = shift_labels.to(logits.device)
       loss = fixed_cross_entropy(logits, shift_labels, None, -100)
       # print(f"pid: {os.getpid()},  LossLayer forward() called")
       return loss

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def get_model_loss_fn(model):
    layers = [TiedLayerSpec(key="embed",typename = PreEmbeddingPipeLayer, model=model),
              *[LayerSpec(DecoderPipeLayer, model=model, layer_idx=idx) for idx in
                range(model.config.num_hidden_layers)],
              LayerSpec(NormPipeLayer, model=model),
              TiedLayerSpec(key="embed", typename = LMHeadLossPipeLayerDummy, model=model),
              ]
    return layers


def get_model(model):
    layers = [TiedLayerSpec(key="embed",typename = PreEmbeddingPipeLayer, model=model),
              *[LayerSpec(DecoderPipeLayer, model=model, layer_idx=idx) for idx in
                range(model.config.num_hidden_layers)],
              LayerSpec(NormPipeLayer, model=model),
              TiedLayerSpec(key="embed", typename = LMHeadLossPipeLayer, model=model),
              ]
    return layers


#obosolete
# def get_model(model):
#     layers = [TiedLayerSpec("word_embeddings", PreEmbeddingPipeLayer, tied_weight_attr='embed_tokens',model=model),
#               *[LayerSpec(DecoderPipeLayer, model=model, layer_idx=idx) for idx in
#                 range(model.config.num_hidden_layers)],
#               LayerSpec(NormPipeLayer, model=model),
#               TiedLayerSpec("word_embeddings", LMHeadPipeLayer, tied_weight_attr='lm_head', model=model),
#               LayerSpec(LossPipeLayer, model=model)]
#     return layers


#no using tied layer, will update both input and output embedding
# def get_model(model):
#     layers = [LayerSpec(PreEmbeddingPipeLayer, model=model),
#               *[LayerSpec(DecoderPipeLayer, model=model, layer_idx=idx) for idx in
#                 range(model.config.num_hidden_layers)],
#               LayerSpec(NormPipeLayer, model=model),
#               LayerSpec(LMHeadPipeLayer, model=model),
#               LayerSpec(LossPipeLayer, model=model)]
#     return layers

class LMHeadLossPipeLayerDummy(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        if is_peft_model(model):
            self.embed_tokens = model.base_model.model.model.embed_tokens
        else:
            self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight
        # print(f"lossDummy memory: {id(self.embed_tokens.weight)}, pid:{os.getpid()}")

    def forward(self, ipt):  #here may not need transmit these variables, reduce memory
        hidden_states, causal_mask = ipt
        # print(f"pid: {os.getpid()},  LMHeadLossPipeLayerDummy() called")
        return hidden_states, causal_mask

class LMHeadLossPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        if is_peft_model(model):
            self.embed_tokens = model.base_model.model.model.embed_tokens
        else:
            self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight

    def forward(self, ipt):
        hidden_states, labels = ipt
        #logits = self.lm_head(hidden_states)
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.shape[-1])
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss(reduction = "mean")
        loss = lce(self.weight, shift_hidden_states, shift_labels)
        # print(f"pid: {os.getpid()},  LMHead forward() called")
        return loss



def loss_fn_parent(model, temperature=1.0,
                   num_iterations=1,
                   gradient_accumulation_steps=1,
                   epsilon_low = 0.0,
                   epsilon_high = 0.0):
    if is_peft_model(model):
        embed_tokens = model.base_model.model.model.embed_tokens
    else:
        embed_tokens = model.model.embed_tokens
    weight = embed_tokens.weight
    #here weight is tied with input embedding matrix, now is lm_head

    def loss_fn(outputs, labels, old_token_logps, global_pipeline_steps, step):
        # print(f"loss_fn memory: id{id(embed_tokens.weight)}, pid:{os.getpid()}")
        hidden_states, causal_mask = outputs
        prompt_completion_ids,logits_to_keep, advantages = labels
        # labels = labels

        # #start of test
        # device_0 = advantages.device
        # advantages = torch.tensor([0.34, -0.1, 0.2, -0.4, 0.9, 0.04, -0.5, -0.7]).to(device_0)
        # #end of test

        original_seq_len = hidden_states.shape[1]
        valid_length = max((causal_mask == 1).sum(dim=-1))

        #shrink
        hidden_states = hidden_states[:, :valid_length, :]
        causal_mask = causal_mask[:,:valid_length]

        prompt_completion_ids  = prompt_completion_ids[:,:valid_length]
        logits_to_keep = logits_to_keep - (original_seq_len - valid_length)

        logits_to_keep = int(logits_to_keep)
        slice_indices = slice(-(logits_to_keep+1), None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :] #1 more position
        slice_indices = slice(-(logits_to_keep), None) if isinstance(logits_to_keep, int) else logits_to_keep
        completion_mask = causal_mask[:, slice_indices]

        # hidden_states = hidden_states[...,:-1,:].contiguous()
        # input_ids = prompt_completion_ids[:, -logits_to_keep:].contiguous()
        #
        # hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # input_ids = input_ids.view(-1)
        #
        # lce = LigerFusedLinearCrossEntropyLoss(reduction="mean")

        #logits = model(input_ids=hidden_states, attention_mask=causal_mask, logits_to_keep=logits_to_keep + 1).logits

        logits = F.linear(hidden_states, weight)
        logits = logits[:, :-1, :]

        input_ids = prompt_completion_ids[:, -logits_to_keep:]

        logits = logits[:, -logits_to_keep:]

        logits = logits / temperature

        per_token_logps =  selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

        if num_iterations > 1:
            if global_pipeline_steps == 0:  #first round in num_iterations rounds
                #use the same, but with grad detached()
                old_per_token_logps  = per_token_logps.detach()
                old_token_logps[step % gradient_accumulation_steps] = old_per_token_logps.clone().detach()  #write back old value to store

            else:  #read old data from buffer
                old_per_token_logps = old_token_logps[step%gradient_accumulation_steps].clone().detach()

            ### wait for complete
        elif num_iterations == 1: #no need share memory for recycle data usage
            old_per_token_logps  = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        #for test :
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        del coef_1, coef_2, per_token_loss,per_token_loss1,per_token_loss2,per_token_logps


        print(f"pid: {os.getpid()},  loss_fn_parent() called, loss: {loss}, advantange: {advantages}")
        return loss
    return loss_fn

def loss_fn_grop_parent(model):
    if is_peft_model(model):
        embed_tokens = model.base_model.model.model.embed_tokens
    else:
        embed_tokens = model.model.embed_tokens
    weight = embed_tokens.weight
    #here weight is tied with input embedding matrix, now is lm_head

    def loss_fn(outputs, labels):
        hidden_states = outputs
        labels = labels
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.shape[-1])
        shift_labels = shift_labels.view(-1)
        lce = LigerFusedLinearCrossEntropyLoss(reduction="mean")
        loss = lce(weight, shift_hidden_states, shift_labels)
        return loss

    return loss_fn
