#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import time
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_utils import seed_worker
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
# from dschat.utils.ds_utils import get_train_ds_config
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)

import deepspeed
# from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator, PipelineModule

from dschat.utils.data.data_utils import create_prompt_dataset_0
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_pipeline_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
# from dschat.utils.perf import print_throughput
# from pipelayers import PreEmbeddingPipeLayer, DecoderPipeLayer, NormPipeLayer, LMHeadPipeLayer, LossPipeLayer
from pipelayers import get_model, DataCollatorForPromptDataset, print_mem

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        # default=['Dahoas/rm-static'],
                        # default = ['lukedai/test'],
                        default = ['open-r1/OpenR1-Math-220k'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='10,0,0',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/ssd/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    # arg1.1
    parser.add_argument('--is_eval',
                        type=bool,
                        default=False,
                        help='whether need to do evaluation')
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader process numer, for both train and eval",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        # default=512,
        default=16384,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        default=True,
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        # default=16,
                        default = 0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_dropout",
                        type=float,
                        default=0.05,
                        help="for dropout for input before sending to lora rank decomposition.")
    parser.add_argument("--lora_alpha",
                        type=float,
                        default=32.0,
                        help="scaling factor for lora matrix muliplication result, the scaling should alpha/r. but implementation may diff")

    parser.add_argument("--lora_module_name",
                        type=str,
                        # default=["layers.0","layers.1","layers.2","layers.3","layers.4","layers.5","layers.6","layers.7","layers.8","layers.9","layers.10","layers.11","layers.12","layers.13","layers.14","layers.15","layers.16","layers.17","layers.18","layers.19","layers.20","layers.21","layers.22","layers.23"],
                        default=[#"layers.0", "layers.1", "layers.2", "layers.3", "layers.4", "layers.5", "layers.6",
                                #"layers.7", "layers.8", "layers.9", "layers.10", "layers.11", "layers.12", "layers.13",
                                #"layers.14", "layers.15", "layers.16", "layers.17", "layers.18", "layers.19",
  #                               "layers.20", "layers.21", "layers.22", "layers.23"
                                 "q_proj","v_proj"
                                 ],

                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        default=True,
                        help='Prints loss at each step.')

    parser.add_argument('--num_stages',
                        default=3,
                        help='pipeline stages.')
    parser.add_argument('--save_model_step',
                        default=2000,
                        help='steps to save model checkpoint.')
    parser.add_argument('--flash_attention',
                        default=True,
                        help='whether using flash attention.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main():

    os.environ["DEEPSPEED_TIMEOUT"] = '100'
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()

    #pipeline ds_config  copy from pipeline chatglm
    ds_config = get_pipeline_ds_config(args)

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier(device_ids=[args.global_rank])
    # torch.distributed.barrier()
    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout,
                            resize_embedding=False,
                            flash_attn = args.flash_attention,
                            dtype=args.dtype)


    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim,lora_scaling=args.lora_alpha, lora_droppout=args.lora_dropout)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

    # Prepare the data
    train_phase = 1

    train_dataset, eval_dataset = create_prompt_dataset_0(
        args.is_eval,
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        sft_only_data_path=args.sft_only_data_path)



    torch.distributed.barrier(device_ids=[args.global_rank])

    #data sampler
    train_sampler = RandomSampler(train_dataset)

    #data collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # data_collator = DataCollatorWithPadding(tokenizer)
    data_collator = DataCollatorForPromptDataset(tokenizer, args.max_seq_len)
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, truncation=True)

    dataloader_params = {
        "batch_size": args.per_device_train_batch_size,
        "collate_fn": data_collator,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": False,
        "sampler": train_sampler,
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "prefetch_factor": None
    }
    train_dataloader = DataLoader(train_dataset, **dataloader_params)

    # pipeline wrap  model
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model_pipe = PipelineModule(layers=get_model(model), num_stages=args.num_stages)
    #here, part of layers has already been moved to cuda:x, others left in cpu, in each process
    model_pipe.to(device).half()

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    ########################################### optimizer and lr scheduler##########################
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model_pipe, args.weight_decay, args.lora_learning_rate) #here needs to change to model_pipe

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))


    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    ################################################################################################

    #pipeline
    engine, _, _, _ = deepspeed.initialize(model=model_pipe,
                                           optimizer=optimizer,
                                           config=ds_config,
                                           model_parameters=model_pipe.parameters(),
                                           lr_scheduler = lr_scheduler,
                                           )

    train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
    # train_dataloader = iter(train_dataloader)
    # Train!
    start = time.time()
    all_loss = 0.0

    print(f"rank: {args.global_rank},  process: {os.getpid()}")

    for step in range(args.num_train_epochs * num_update_steps_per_epoch-1):  #-1 is importtant , abandon last residual to avoid error
        start1 = time.time()
        print_rank_0(
            f"step {step}, progress: {(step*1.0)/(args.num_train_epochs * num_update_steps_per_epoch)}", args.global_rank)

        loss = engine.train_batch(data_iter=train_dataloader)
        end1 = time.time()
        # torch.cuda.empty_cache()   #clear cache, for test sd
        if args.print_loss:
            print(
                f"step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}, time comsumed = {end1-start1}"
            )
#check mem
        # allocated = torch.cuda.memory_allocated()  # allocated mem
        # max_allocated = torch.cuda.max_memory_allocated()  # maximum allocated mem in history
        # reserved = torch.cuda.memory_reserved()  # allocated + cache mem
        # max_reserved = torch.cuda.max_memory_reserved()  # maximum allocated + cache mem in history
        #
        # print(f"step: {step}, Rank: {torch.distributed.get_rank()}\n"
        #       f"Global Rank {args.global_rank} ) - "
        #       f"Total memory: {total / (1024 ** 2)} MB, "
        #       f"Allocated: {allocated / 1024 ** 2:.2f} MB, "
        #       f"Max Allocated: {max_allocated / 1024 ** 2:.2f} MB, "
        #       f"Reserved: {reserved / 1024 ** 2:.2f} MB, "
        #       f"Max Reserved: {max_reserved / 1024 ** 2:.2f} MB"
        #       )

        if (step + 1) % args.save_model_step == 0:
            print(f"Saving at step {step}")
            engine.save_checkpoint(args.output_dir)
            if args.global_rank == 0:
                tokenizer.save_vocabulary(args.output_dir)
                CONFIG_NAME = "config.json"
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                model.config.to_json_file(output_config_file)

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        engine.save_checkpoint(args.output_dir)

    torch.distributed.barrier(device_ids=[args.global_rank])
    print(f"finished saving model")

    if args.global_rank == 0:
        tokenizer.save_vocabulary(args.output_dir)
        CONFIG_NAME = "config.json"
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        model.config.to_json_file(output_config_file)
        print(f"finished save vocabulary config and model config")
    torch.distributed.barrier(device_ids=[args.global_rank])
    print(f"done after sync, will exit programm ")


    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return

if __name__ == "__main__":
    main()
