'''
this code is to split qwen2.5b-1.5-instruct to multiple classes, for parallel processing
'''
import torch
from deepspeed.runtime.pipe import TiedLayerSpec, LayerSpec
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from transformers import Qwen2ForCausalLM, Qwen2Model
import torch.nn as nn
import os

# from inference.huggingface.zero_inference.utils import hidden_bytes

'''
1st layer, input: input_ids,
output:hidden_states, position_ids, 
'''

#this version ,trys to pass only one data between layers
#try to generate cos, sin by layer itself. but the issue is\
#each layer still need to store rotary embed as buffer
#so may occupy more space
class PreEmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight
        self.rotary_emb = model.model.rotary_emb
        # self.weight = self.embed_tokens.weight

    def forward(self,  ipt):
        input_ids, labels = ipt
        inputs_embeds = self.embed_tokens(input_ids) #[bz, seq_len] -> [bz, seq_len, hidden_v]

        hidden_states = inputs_embeds   #[bz, seq_len, hidden_v]


        return hidden_states

class DecoderPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM, layer_idx:int):
        super().__init__()
        self.layer = model.model.layers[layer_idx]
        self.layer_idx = torch.tensor(layer_idx)

    def forward(self, ipt):
        # print(f"decoder layer: {self.layer_idx} been called")
        hidden_states= ipt
        cache_position = torch.arange(
            0, 0 + hidden_states.shape[1], device=hidden_states.device
        )
        position_ids = cache_position.unsqueeze(0)  # [1 ,  seq_len]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_outputs = self.layer(hidden_states,
                                   position_ids = position_ids,
                                   cache_position = cache_position,
                                   position_embeddings = position_embeddings)

        hidden_states = layer_outputs[0]
        # print(f"pid: {os.getpid()},  DecoderLayer.{self.layer_idx} forward() called")
        return hidden_states

class NormPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        self.norm = model.model.norm

    def forward(self, ipt):
        hidden_states = ipt
        hidden_states = self.norm(hidden_states)
        # print(f"pid: {os.getpid()},  NormLayer forward() called")
        return hidden_states

class LMHeadPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight

    def forward(self, ipt):
        hidden_states = ipt
        #logits = self.lm_head(hidden_states)
        logits = torch.nn.functional.linear(hidden_states, self.embed_tokens.weight)
        # print(f"pid: {os.getpid()},  LMHead forward() called")
        return logits

# class LossPipeLayer(torch.nn.Module):
#     def __init__(self, model: Qwen2ForCausalLM):
#         super().__init__()
#
#     def forward(self, ipt):
#        logits, labels = ipt
#        # loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
#        # Upcast to float if we need to compute the loss to avoid potential precision issues
#        logits = logits.float()
#        labels = labels.to(logits.device)
#        # Shift so that tokens < n predict n
#        labels = nn.functional.pad(labels, (0, 1), value=-100)
#        shift_labels = labels[..., 1:].contiguous()
#
#        # Flatten the tokens
#        vocab_size = logits.shape[-1]
#        logits = logits.view(-1, vocab_size)
#        shift_labels = shift_labels.view(-1)
#        # Enable model parallelism
#        shift_labels = shift_labels.to(logits.device)
#        loss = fixed_cross_entropy(logits, shift_labels, None, -100)
#        # print(f"pid: {os.getpid()},  LossLayer forward() called")
#        return loss

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


#just pass input to output, do nothing, as there is customized loss_fn
class LMHeadLossPipeLayerDummy(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight

    def forward(self, ipt):
        hidden_states = ipt
        return hidden_states


#for using liger, combine lmhead layer and loss layer together, no loss_fn
class LMHeadLossPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
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
#


def loss_fn_parent(model):
    embed_tokens = model.model.embed_tokens
    weight = embed_tokens.weight

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
