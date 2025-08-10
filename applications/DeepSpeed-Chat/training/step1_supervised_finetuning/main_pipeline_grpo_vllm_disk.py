#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import time
import os
import shutil
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from trl.trainer.grpo_trainer import RepeatRandomSampler
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
    DataCollatorForLanguageModeling,
    AutoConfig
)

import deepspeed
# from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator, PipelineModule

from dschat.utils.data.data_utils import create_prompt_dataset_grpo, create_prompt_dataset_0
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_pipeline_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
# from dschat.utils.perf import print_throughput
# from pipelayers import PreEmbeddingPipeLayer, DecoderPipeLayer, NormPipeLayer, LMHeadPipeLayer, LossPipeLayer
from pipelayers_grpo import get_model,get_model_loss_fn, DataCollatorForPromptDataset,DataCollatorForPromptDatasetDummy, print_mem, loss_fn_parent, loss_fn_parent_liger,loss_fn_parent_policy_gradient
from grpo import get_reward_funcs, enable_gradient_checkpointing,check_module_requires_grad,PipelineGRPOEngine
from peft import LoraConfig, PeftConfig, get_peft_model
from accelerate.utils import is_peft_model
from pipelayers_grpo import (convert_model_to_hf_qwen25_500m,
                             test_load_model,
                             convert_model_to_hf_qwen25_3b,
                             convert_model_to_hf_qwen25_3b_no_bin,
                             convert_model_to_hf_qwen25_500m_no_bin,
                             convert_model_to_hf_deepseek_1500m_no_bin,
                             TestSampler)
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from pathlib import Path

#for liger-kernel loss
from transformers.utils.import_utils import _is_package_available
from packaging import version
LIGER_KERNEL_MIN_VERSION = "0.5.6"
_is_liger_kernel_available, _liger_kernel_version = _is_package_available("liger_kernel", return_version=True)
def is_liger_kernel_available(min_version: str = LIGER_KERNEL_MIN_VERSION) -> bool:
    return _is_liger_kernel_available and version.parse(_liger_kernel_version) >= version.parse(min_version)

from ligerloss import LigerFusedLinearGRPOLoss
#end of liger kernel

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        # default=['Dahoas/rm-static'],
                        # default = ['lukedai/test'],
                        # default = ['open-r1/OpenR1-Math-220k'],
                        default = ["ricdomolm/MATH-500"],
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
        default='/ssd2/tmp/data_files/',
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
        # default="Qwen/Qwen2.5-3B-Instruct",
        # default="lukedai/qwen2.5-3b-sft",
        default="lukedai/model_deepseek_1.5b",
        # default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7b",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
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
        default=1e-3,
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
        default=4,
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
                        default=True,
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--use_reentrant',
                        default=False,
                        help='use_reentrant for gradient checkpointing for model.')


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
    parser.add_argument('--torch_dtype',
                        type=str,
                        default='bfloat16',
                        choices=['bfloat16', 'float16'],
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
        default=1e-3,
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
                        default=5,
                        help='pipeline stages.')
    parser.add_argument('--save_model_step',
                        default=1,
                        help='steps to save model checkpoint. should be 1 only')
    parser.add_argument('--flash_attention',
                        default="flash_attention_2",
                        help='whether using flash attention.')
    parser.add_argument('--use_liger_kernel',
                        default=True,
                        help='whether using liger kenel.')
    parser.add_argument('--custom_loss_fn',
                        default=True,
                        help='whether using loss_fn for last stage.')

    parser.add_argument('--reward_funcs',
                        # default=['accuracy','format','tag_count'],
                        default=['accuracy'],
                        help='reward functions for reinforcement learning.')

    parser.add_argument('--reward_weights',
                        # default=[1.0, 1.0, 1.0],
                        default=[1.0],
                        help='reward functions weights for reinforcement learning.')

    parser.add_argument('--num_generations',
                        default=4,
                        help='grpo G, number of generations for each prompt')

    parser.add_argument('--max_prompt_length',
                        default=512,
                        help='maximum length of prompt')

    parser.add_argument('--repetition_penalty',
                        default=1.0,
                        help='vllm parameter for penalty for repetiton')

    parser.add_argument('--temperature',
                        default=1.0,
                        help='vllm parameter ')

    parser.add_argument('--top_p',
                        default=1.0,
                        help='vllm parameter ')

    parser.add_argument('--top_k',
                        default=-1,
                        help='vllm parameter ')

    parser.add_argument('--min_p',
                        default=0.0,
                        help='vllm parameter ')

    parser.add_argument('--max_completion_length',
                        default=1024,
                        help='vllm parameter ')


    parser.add_argument('--num_iterations',
                        default=1,
                        help='repeat count for each batch data sent to training ')

    parser.add_argument('--grpo_beta',
                        default=0.0,
                        help='beta for kl distance penalty of per_token_logps and ref_per_token_logps')

    parser.add_argument('--epsilon_low',
                        default=0.1,
                        help='clap lower boundary for grpo loss')

    parser.add_argument('--epsilon_high',
                        default=0.1,
                        help='clap higher boundary for grpo loss')

    parser.add_argument('--use_vllm',
                        default=True,
                        help='whether use vllm server to generate')



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

    # tokenizer_model_name = "Qwen/Qwen2.5-3B-Instruct" if args.model_name_or_path == "lukedai/qwen2.5-3b-sft" else args.model_name_or_path

    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=False,
                                  add_special_tokens=additional_special_tokens)



#for test

    torch_dtype = (
        args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
    )
    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(args)
    reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)





    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout,
                            resize_embedding=False,
                            attn_implementation = args.flash_attention,
                            torch_dtype=torch_dtype,
                            use_liger_kernel=args.use_liger_kernel,
                            gradient_checkpointing = args.gradient_checkpointing)

    #for save usage
    base_config = AutoConfig.from_pretrained(args.model_name_or_path)



    if args.global_rank == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, param size: {param.data.shape}")
            # pass
        print(model)

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)

    ########### lora
    # if args.lora_dim > 0:
    #     model = convert_linear_layer_to_lora(model, args.lora_module_name,
    #                                          args.lora_dim,lora_scaling=args.lora_alpha, lora_droppout=args.lora_dropout)
    #     if args.only_optimize_lora:
    #         model = only_optimize_lora_parameters(model)
    #         model = make_model_gradient_checkpointing_compatible(model)

    peft_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=args.lora_dim,
        target_modules=args.lora_module_name,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    if args.lora_dim > 0:
        model = get_peft_model(model, peft_config)

    if args.global_rank == 0:
        # print(model)
        pass

    # test for connecting to vllm server
    model = model.to(device)  #only for test, will use deepspeed.initialize() instead
    vllm_client = None
    print(f"rank: {args.global_rank}")
    if args.global_rank == 0 and args.use_vllm :
        from trl.extras.vllm_client import VLLMClient
        from vllm import SamplingParams

        #for test, comment it,
        vllm_client = VLLMClient(
            '0.0.0.0', 8000, connection_timeout=1200.0
        )
        vllm_client.init_communicator()
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        responses = vllm_client.generate(prompts=prompts, n=4, max_tokens=32,
                                         )
        responses_txt = tokenizer.batch_decode(responses)
        print("Test vllm Server Responses:", responses_txt)  # noqa

    test_updating = True  #temprary shutdown
    if test_updating and args.global_rank == 0 and args.use_vllm:
        # test for updating model parameter
        if is_peft_model(model):
            model.merge_adapter()
            for name, param in model.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if model.prefix in name:
                    continue
                if "original_module" in name:
                    continue
                name = name.replace("modules_to_save.default.", "")
                if args.global_rank == 0:
                    vllm_client.update_named_param(name, param.data)
                    # print(f"update param name: {name}, param size: {param.data.shape}")
            model.unmerge_adapter()
        else:
        # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in model.named_parameters():
                if args.global_rank == 0:
                    # print(f"name: {name}")
                    vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if args.global_rank==0:
            vllm_client.reset_prefix_cache()
        #########################end of updating weight####################
    # return
    # print(model)
    if args.global_rank == 0:
        # check_module_requires_grad(model)
        pass

    #gradient_checkpointing
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model = enable_gradient_checkpointing(model, args)

    # return
    # Prepare the data

    train_phase = 1

##for grpo, need _grpo func, not _0 func
    train_dataset, eval_dataset = create_prompt_dataset_grpo(
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




    #for grpo reward debug purpose
    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("messages")

    torch.distributed.barrier(device_ids=[args.global_rank])

    #data sampler
    # train_sampler = RandomSampler(train_dataset)
    # there is possibility that batch_size_sampler<1,

    batch_size_sampler = args.per_device_train_batch_size * args.gradient_accumulation_steps // args.num_generations
    assert batch_size_sampler > 0

    # train_sampler = RepeatRandomSampler(data_source=train_dataset,
    #                                     mini_repeat_count=args.num_generations,
    #                                     batch_size=batch_size_sampler,
    #                                     repeat_count=args.num_iterations,  #
    #                                     seed=123)  #here need to consider seed?

    train_sampler = TestSampler(data_source=train_dataset,
                                        mini_repeat_count=args.num_generations,
                                        batch_size=batch_size_sampler,
                                        repeat_count=args.num_iterations,  #
                                        seed=123)  #here need to consider seed?



# ######################### old data collator for sft   ##
#     #data collator
#     # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#     # data_collator = DataCollatorWithPadding(tokenizer)
#     if args.custom_loss_fn:
#         data_collator = DataCollatorForPromptDatasetDummy(tokenizer, args.max_seq_len)
#     else:
#         data_collator = DataCollatorForPromptDataset(tokenizer, args.max_seq_len)
#     #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, truncation=True)
# #############################################################
    # Data collator for grpo
    def data_collator(features):  # No data collation is needed in GRPO
        return features

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
    # if args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    # print(model)
    #loss = loss_fn(outputs, label)
    if args.custom_loss_fn:

        if args.use_liger_kernel:
            liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=0,
                epsilon_low=args.epsilon_low,
                epsilon_high=args.epsilon_high,
                temperature=args.temperature,
                use_ref_model=False,
            )

        model_pipe = PipelineModule(layers=get_model_loss_fn(model, is_tied_embedding=base_config.tie_word_embeddings),
                                    num_stages=int(args.num_stages),
                                    # activation_checkpoint_interval = 4
                                    # loss_fn=loss_fn_parent_liger(model,
                                    #                        num_iterations=args.num_iterations,
                                    #                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    #                        liger_loss = liger_grpo_loss,
                                    #                        is_tied= base_config.tie_word_embeddings
                                    #                        ) if args.use_liger_kernel
                                    #
                                    loss_fn = loss_fn_parent_policy_gradient(model,
                                                                             temperature=args.temperature,
                                                                             num_iterations = args.num_iterations,
                                                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                                             epsilon_low = args.epsilon_low,
                                                                             epsilon_high = args.epsilon_high,
                                                                             is_tied = base_config.tie_word_embeddings
                                                                            ) if args.use_liger_kernel
                                    else loss_fn_parent(model,
                                                           temperature=args.temperature,
                                                           num_iterations=args.num_iterations,
                                                           gradient_accumulation_steps=args.gradient_accumulation_steps,
                                                           epsilon_low=args.epsilon_low,
                                                           epsilon_high=args.epsilon_high,
                                                           )
                                    )
    else:
        model_pipe = PipelineModule(layers=get_model(model),
                                    num_stages=int(args.num_stages),
                                    # activation_checkpoint_interval = 4
                                    )
    #here, part of layers has already been moved to cuda:x, others left in cpu, in each process
    # model_pipe.to(device).half()

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    print_rank_0(
        f"num_update_steps_per_epoch: {num_update_steps_per_epoch}", args.global_rank)
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
    pipeline_grpo_config = {'pipeline_grpo':True,
                            'engine': PipelineGRPOEngine,
                            'tokenizer': tokenizer,
                            'max_prompt_length':args.max_prompt_length,
                            'vllm_client':vllm_client,
                            'use_vllm':args.use_vllm,
                            'num_generations':args.num_generations,
                            'repetition_penalty' : args.repetition_penalty,
                            'temperature' : args.temperature,
                            'top_p' : args.top_p,
                            'top_k' : args.top_k,
                            'min_p' : args.min_p,
                            'max_completion_length': args.max_completion_length ,
                            'num_iterations':args.num_iterations,
                            'grpo_beta':args.grpo_beta,
                            'reward_funcs':reward_funcs,
                            'reward_weights':reward_weights}

    engine, _, _, _ = deepspeed.initialize(model=model_pipe,
                                           optimizer=optimizer,
                                           config=ds_config,
                                           model_parameters=model_pipe.parameters(),
                                           lr_scheduler = lr_scheduler,
                                           pipeline_grpo_config=pipeline_grpo_config
                                           )


    train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
    # train_dataloader = iter(train_dataloader)
    # Train!
    start = time.time()
    all_loss = 0.0

    print_mem(args.global_rank, device)
    #clear cache of cuda
    torch.cuda.empty_cache()

    for step in range(args.num_train_epochs * num_update_steps_per_epoch-1):  #-1 is importtant , abandon last residual to avoid error
    # for step in range(1):
        torch.cuda.empty_cache()
        #batch = next(train_dataloader)
        start1 = time.time()

        print_rank_0(
            f"step {step}, progress: {(step*1.0)/(args.num_train_epochs * num_update_steps_per_epoch)}", args.global_rank)

        print_mem(args.global_rank, device, info=f"before step:{step} training")

        loss = engine.train_batch(data_iter = train_dataloader,
                                  tokenizer=tokenizer)

        end1 = time.time()
        # torch.cuda.empty_cache()   #clear cache, for test sd
        if args.print_loss:
            print(
                f"step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}, time comsumed = {end1-start1}"
            )
#check mem
        # print_mem(torch.distributed.get_rank(), device, f"after step {step} of training:")
        if (step + 1) % args.save_model_step == 0:
            if args.global_rank == 0:
                #remove previous saving folder, only process 0
                pre_tag = f"global_step{engine.global_steps - args.save_model_step}"
                new_tag = f"global_step{engine.global_steps}"
                existing_folder = os.path.join(args.output_dir, pre_tag)
                new_folder = os.path.join(args.output_dir, new_tag)
                if os.path.isdir(existing_folder):

                    # #start of copy 1 file to check whether weight changes
                    # if (step + 1) % 4 == 0:
                    #     save_file = os.path.join(existing_folder,'layer_24-model_states.pt')
                    #     new_index_tag = f"step_{step+1}"
                    #     new_name = os.path.join(args.output_dir, new_index_tag)
                    #     shutil.copy(save_file, new_name)
                    # #end of copy 1 file to check whether weight changes

                    shutil.rmtree(existing_folder)
                    print(f"remove folder {existing_folder}")

            #each process all saving weight
            print(f"Saving at step {step}")
            engine.save_checkpoint(args.output_dir)

            #only for process 0 and at the begining of training
            if args.global_rank == 0 and engine.global_steps <= args.save_model_step:
                tokenizer.save_pretrained(args.output_dir)
                base_config.save_pretrained(args.output_dir)
                model.generation_config.save_pretrained(args.output_dir)

            # COLLECT_PARAMS = False
            # if COLLECT_PARAMS:
            #     #start of saving params directly ,replacing engine.save_checkpoint()
            #     if args.global_rank >= 0:
            #         module = engine.module
            #         num_layers = len(module.forward_funcs)
            #         start, end = 0, num_layers
            #         layer_list = module.forward_funcs[start:end]
            #         model_static_dict = {}
            #         for idx, layer in enumerate(layer_list):
            #             model_ckpt_path = module.ckpt_layer_path(ckpt_dir='/tmp',local_layer_idx=start + idx)
            #             layer_i = int(model_ckpt_path.split('/')[-1].split('-')[0].replace('layer_', ''))
            #             orig_state_dict = layer.state_dict()
            #             final_state_dict = clone_tensors_for_torch_save(orig_state_dict)
            #             print("已经处理layer：{}".format(layer_i))
            #             if layer_i == 0:
            #                 model_static_dict["model.embed_tokens.weight"] = final_state_dict["embed_tokens.weight"]
            #             elif layer_i <= 24 and layer_i >= 1:
            #                 for k, v in final_state_dict.items():
            #                     model_static_dict["model." + k.replace("layer.", "layers.{}.".format(layer_i - 1), 1)] = v
            #             elif layer_i == 25:  # norm layer
            #                 model_static_dict['model.norm.weight'] = final_state_dict['norm.weight']
            #             elif layer_i == 26:
            #                 model_static_dict["lm_head.weight"] = final_state_dict["embed_tokens.weight"]
            #         #each process has its own layer in model_static_dict
            #         #need to collect them into process 0
            #     if args.global_rank == 0 and args.use_vllm:
            #         for k, v in model_static_dict.items():
            #             v = v.to(device)
            #             vllm_client.update_named_param(k, v.data)
            #
            #     #the problem here:
            #     # 1. vllm server can only receive from one device
            #     # mutiple stages params needs to first gather to one device,
            #     # waste memory
            #     #
            #     # this version is not working

            #process 0 read from separate files and update to vllm server
            WRITE_TO_DISK=True
            # if args.global_rank == 0 and args.use_vllm and WRITE_TO_DISK:
            #     #read state dict data of each layer from disk files, and save into bin
            #     convert_model_to_hf_qwen25_500m(new_folder, args.output_dir)
            #     model, tokenizer = test_load_model(args.output_dir)
            #     model = model.to(device)
            #     for name, param in model.named_parameters():
            #         # print(f"name: {name}")
            #         vllm_client.update_named_param(name, param.data)
            #     # Reset cache on main process
            #     vllm_client.reset_prefix_cache()
            if args.global_rank == 0 and args.use_vllm and WRITE_TO_DISK:
                #read state dict data of each layer from disk files, and save into pytorch.bin file
                # convert_model_to_hf_qwen25_500m(new_folder, args.output_dir)
                model_static_dict = convert_model_to_hf_deepseek_1500m_no_bin(new_folder)
                #
                # test_result = model_static_dict.get('model.layers.0.self_attn.q_proj.weight')[0, :4]
                # print(f"test result: {test_result}")
                counttt = 0
                for name, param in model.named_parameters():
                    # print(f"name: {name}")
                    counttt += 1
                    new_param = model_static_dict.get(name).to(device)
                    # print(f"[upload] name: {name}, size: {param.shape}")
                    vllm_client.update_named_param(name, new_param)
                # Reset cache on main process
                vllm_client.reset_prefix_cache()
            elif args.global_rank == 0 and args.use_vllm and WRITE_TO_DISK==False:
                # convert_model_to_hf(new_folder, args.output_dir)
                pass

    # torch.cuda.empty_cache()
        # print_mem(args.global_rank, device, info=f"after step:{step} training")

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        engine.save_checkpoint(args.output_dir)

    torch.distributed.barrier(device_ids=[args.global_rank])
    print(f"finished saving model")

    if args.global_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        base_config.save_pretrained(args.output_dir)
        model.generation_config.save_pretrained(args.output_dir)
        print(f"finished save vocabulary config and model config")
    torch.distributed.barrier(device_ids=[args.global_rank])
    print(f"done after sync, will exit programm ")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return

if __name__ == "__main__":
    main()
