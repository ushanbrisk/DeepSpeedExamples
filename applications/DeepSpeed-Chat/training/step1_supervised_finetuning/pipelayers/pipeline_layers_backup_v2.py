'''
this code is to split qwen2.5b-1.5-instruct to multiple classes, for parallel processing
'''
import torch
from deepspeed.runtime.pipe import TiedLayerSpec, LayerSpec
from training.step1_supervised_finetuning.ligerloss import LigerFusedLinearCrossEntropyLoss

from transformers import Qwen2ForCausalLM, Qwen2Model
import torch.nn as nn
import os
from torch.utils.checkpoint import checkpoint
# from inference.huggingface.zero_inference.utils import hidden_bytes
from training.step1_supervised_finetuning.grpo import hash_tensor
from trl.trainer.utils import selective_log_softmax

'''
1st layer, input: input_ids,
output:hidden_states, position_ids, 
'''
class PreEmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight
        self.rotary_emb = model.model.rotary_emb
        # self.weight = self.embed_tokens.weight
        # print(f"PreEmbedding: dtype {self.weight.data.dtype}")

    def forward(self,  ipt):
        print(f"Embedding:{hash_tensor(self.weight.data)}")

        input_ids, labels = ipt
        inputs_embeds = self.embed_tokens(input_ids) #[bz, seq_len] -> [bz, seq_len, hidden_v]
        # print(f"PreEmbedding2: dtype {self.embed_tokens.weight.data.dtype}")
        cache_position = torch.arange(
            0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)  #[1 ,  seq_len]
        # causal_mask = self._update_causal_mask(
        #             attention_mask, inputs_embeds, cache_position, past_key_values=None, output_attentions=False
        # )
        causal_mask = None
        hidden_states = inputs_embeds   #[bz, seq_len, hidden_v]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings
        requires_grad_idx = torch.tensor([3]).to(hidden_states.device)  #here 3 different from [3], 3 may cause error in communication
        # print(f"pid: {os.getpid()},  PreEmbedding forward() called")
        return requires_grad_idx, cos, sin, hidden_states, position_ids, cache_position, labels

class DecoderPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM, layer_idx:int):
        super().__init__()
        self.layer = model.model.layers[layer_idx]
        self.layer_idx = torch.tensor(layer_idx)

    def forward(self, ipt):
        # print(f"decoder layer: {self.layer_idx} been called")
        # print(f"decoder layer {self.layer_idx}: weight hash:{hash_tensor(self.layer.self_attn.q_proj.weight)}")
        requires_grad_idx, cos, sin, hidden_states, position_ids, cache_position, labels = ipt
        if cos.requires_grad == True:
            cos_ = cos.detach()
            del cos
            cos = cos_
            del cos_
            # print(f"pid: {os.getpid()},  DecoderLayer.{self.layer_idx} changed cos grad() called")
        if sin.requires_grad == True:
            sin_ = sin.detach()
            del sin
            sin = sin_
            del sin_


        position_embeddings = (cos, sin)

        #for test
        layer_outputs = checkpoint(self.layer, hidden_states,position_ids = position_ids,
                                   cache_position = cache_position,
                                   position_embeddings = position_embeddings,
                                   use_reentrant = False)

        # layer_outputs = self.layer(hidden_states,
        #                            position_ids = position_ids,
        #                            cache_position = cache_position,
        #                            position_embeddings = position_embeddings)

        hidden_states = layer_outputs[0]
        # print(f"pid: {os.getpid()},  DecoderLayer.{self.layer_idx} forward() called")
        return requires_grad_idx, cos, sin, hidden_states, position_ids, cache_position, labels

class NormPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        self.norm = model.model.norm

    def forward(self, ipt):
        requires_grad_idx, cos, sin, hidden_states, position_ids, cache_position, labels = ipt
        # hidden_states = self.norm(hidden_states)
        hidden_states = checkpoint(self.norm, hidden_states, use_reentrant = False)
        # print(f"pid: {os.getpid()},  NormLayer forward() called")
        return hidden_states, labels

class LMHeadPipeLayer(torch.nn.Module):
    def __init__(self, model:Qwen2ForCausalLM):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight

    def forward(self, ipt):
        hidden_states, labels = ipt
        #logits = self.lm_head(hidden_states)
        logits = torch.nn.functional.linear(hidden_states, self.embed_tokens.weight)
        # print(f"pid: {os.getpid()},  LMHead forward() called")
        return logits, labels

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
        self.embed_tokens = model.model.embed_tokens
        self.weight = self.embed_tokens.weight
        # self.idle = True    #False when for ref model usage
        # self.temperature = 1.0
        # vocab_size, hid_size = self.weight.shape
        # self.lm_head = nn.Linear(hid_size, vocab_size, bias=False)
        # self.lm_head.weight = self.weight

    # def set_idle_ref_model(self):
    #     self.idle = False

    def forward(self, ipt):
        hidden_states, labels = ipt
        return hidden_states
        # if self.idle:
        #     return hidden_states
        # else:
        #
        #     # per_token_kl = (
        #     #         torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        #     # )
        #
        #     # logits = self.lm_head(hidden_states[:, slice_indices, :])
        #
        #     batch_size = 1
        #     all_logps = []
        #     all_entropies = []
        #     for start in range(0, hidden_states.shape[0], batch_size):
        #         row_hidden_states = hidden_states[start:start+batch_size]
        #         row_labels = labels[start:start+batch_size]
        #         row_logits = self.lm_head(row_hidden_states)
        #         row_logits = row_logits / self.temperature
        #         row_logits = row_logits[..., :-1, :].contiguous()
        #         row_shift_labels = row_labels[..., 1:].contiguous()
        #         per_token_logps = selective_log_softmax(row_logits, row_shift_labels)
        #         all_logps.append(per_token_logps)
        #     logps = torch.cat(all_logps, dim=0)
        #     return logps
        #




            #
            # logits = self.lm_head(hidden_states)
            #
            # logits = logits / self.temperature
            #
            #
            # logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            #
            # per_token_logps = selective_log_softmax(logits, shift_labels)

            # return per_token_logps

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

def loss_fn_parent_ref_vanilla(model, ref_module):
    embed_tokens = model.model.embed_tokens
    weight = embed_tokens.weight
    ref_lm_head = ref_module
    temperature = 1.0

    def loss_fn(outputs, labels, ref_hidden=None):
        # if ref_hidden is not None:
        #     print(f"loss_fn, outputs:{hash_tensor(outputs)}, outputs:{outputs[0][1][:5]}, ref_outputs:{hash_tensor(ref_hidden)}, ref_outputs:{ref_hidden[0][1][:5]}")

        print(f"loss_fn,hid:{hash_tensor(outputs)}, ref_hid:{hash_tensor(ref_hidden)}, lm_head weight:{hash_tensor(weight.data)}, ref_lm_head weight:{hash_tensor(ref_lm_head.weight.data)}")

        train_hidden_states = outputs
        labels = labels
        shift_hidden_states = train_hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.shape[-1])
        # shift_labels = shift_labels.view(-1)

        ref_hidden = ref_hidden[..., :-1, :].contiguous()

        all_logps = []
        all_entropies = []
        batch_size = 1
        for start in range(0, ref_hidden.size(0),batch_size ):
            ref_hidden_batch = ref_hidden[start:start+batch_size]
            ref_label_batch = shift_labels[start:start+batch_size]
            ref_logits_batch = ref_lm_head(ref_hidden_batch)
            # ref_logits_batch = ref_logits_batch[:,:-1,:]
            ref_logits_batch = ref_logits_batch/temperature
            logps = selective_log_softmax(ref_logits_batch, ref_label_batch)
            all_logps.append(logps)
        logps = torch.cat(all_logps, dim=0)
        return logps
        # lce = LigerFusedLinearCrossEntropyLoss(reduction="mean")
        # loss = lce(weight, shift_hidden_states, shift_labels)
        # return loss

    return loss_fn

#original func, for no ref model usage
def loss_fn_parent_no_ref(model):
    embed_tokens = model.model.embed_tokens
    weight = embed_tokens.weight

    def loss_fn(outputs, labels):
        # if ref_hidden is not None:
        #     print(f"loss_fn, outputs:{hash_tensor(outputs)}, outputs:{outputs[0][1][:5]}, ref_outputs:{hash_tensor(ref_hidden)}, ref_outputs:{ref_hidden[0][1][:5]}")

        print(f"loss_fn,hid:{hash_tensor(outputs)}, , lm_head weight:{hash_tensor(weight.data)}")

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



def loss_fn_parent(model, ref_module):
    embed_tokens = model.model.embed_tokens
    weight = embed_tokens.weight
    ref_lm_head = ref_module
    temperature = 1.0

    def loss_fn(outputs, labels, ref_hidden=None):
        # if ref_hidden is not None:
        #     print(f"loss_fn, outputs:{hash_tensor(outputs)}, outputs:{outputs[0][1][:5]}, ref_outputs:{hash_tensor(ref_hidden)}, ref_outputs:{ref_hidden[0][1][:5]}")

        print(f"loss_fn,hid:{hash_tensor(outputs)}, ref_hid:{hash_tensor(ref_hidden)}, lm_head weight:{hash_tensor(weight.data)}, ref_lm_head weight:{hash_tensor(ref_lm_head.weight.data)}")

        train_hidden_states = outputs
        labels = labels
        shift_hidden_states = train_hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.shape[-1])
        # shift_labels = shift_labels.view(-1)

        ref_hidden = ref_hidden[..., :-1, :].contiguous()

        all_logps = []
        all_entropies = []
        batch_size = 1
        shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.shape[-1])
        shift_labels = shift_labels.view(-1)
        ref_hidden = ref_hidden.view(-1, ref_hidden.shape[-1])
        # for start in range(0, ref_hidden.size(0),batch_size ):
        #     ref_hidden_batch = ref_hidden[start:start+batch_size]
        #     ref_label_batch = shift_labels[start:start+batch_size]
        #     ref_logits_batch = ref_lm_head(ref_hidden_batch)
        #     # ref_logits_batch = ref_logits_batch[:,:-1,:]
        #     ref_logits_batch = ref_logits_batch/temperature
        #     logps = selective_log_softmax(ref_logits_batch, ref_label_batch)
        #     all_logps.append(logps)
        # logps = torch.cat(all_logps, dim=0)
        # return logps
        lce = LigerFusedLinearCrossEntropyLoss(beta=1.0, reduction="mean")
        loss = lce(weight, shift_hidden_states, shift_labels, ref_hidden, ref_lm_head.weight)
        # loss = lce(weight, shift_hidden_states, shift_labels)
        return loss

    return loss_fn
