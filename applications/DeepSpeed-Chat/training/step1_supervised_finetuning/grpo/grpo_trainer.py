
import os
import textwrap
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sized, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

if is_peft_available():
    from peft import PeftConfig, get_peft_model

def enable_gradient_checkpointing(model: PreTrainedModel, args) -> PreTrainedModel:
    """Enables gradient checkpointing for the model."""
    # Ensure use_cache is disabled
    model.config.use_cache = False

    gradient_checkpointing_kwargs={'use_reentrant': False}

    # Enable gradient checkpointing on the base model for PEFT
    if is_peft_model(model):
        model.base_model.gradient_checkpointing_enable()  #here need to reconsider
    # Enable gradient checkpointing for non-PEFT models
    else:
        model.gradient_checkpointing_enable()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    #not set:true
    #set, true->true,    false->false

    if args.use_reentrant:
        model.enable_input_require_grads()


    return model


def check_module_requires_grad(model):
    for name, module in model.named_modules():
        params_require_grad = any(p.requires_grad for p in module.parameters(recurse=False))
        if params_require_grad:
            print(f"[need grad] module name: {name} | type: {type(module).__name__}")
        else:
            print(f"[freeze] module name: {name} | type: {type(module).__name__}")

#
# check_module_requires_grad(model)

