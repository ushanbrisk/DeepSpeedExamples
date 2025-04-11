# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, )

from dschat.utils.model.model_utils import create_hf_model
from dschat.utils.utils import load_hf_tokenizer
from deepspeed import get_accelerator
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        # default="Qwen/Qwen2.5-0.5B-Instruct",
        # default="/home/luke/distributed_machine_learning/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_test_202503131848",
        # default="/home/luke/distributed_machine_learning/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_test_202503162329",
        default = "lukedai/Qwen2.5-1.5B-Open-R1-Distill-sft-v2",
        # default = "Qwen/Qwen2.5-1.5B-Instruct",
        help="Path to baseline model",
        # required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        default="/home/luke/distributed_machine_learning/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/output_test",
        help="Path to pretrained model",
        # required=True,
    )


    parser.add_argument(
        "--model_name_or_path_finetuneC",
        type=str,
        # default="lukedai/Qwen2.5-1.5B-Open-R1-Distill",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Path to pretrained model",
        # required=True,
    )



    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=2048
    )
    # generate_ids = model.generate(inputs.input_ids,
    #                               num_beams=num_beams,
    #                               num_beam_groups=num_beam_groups,
    #                               do_sample=do_sample,
    #                               num_return_sequences=num_return_sequences,
    #                               max_new_tokens=max_new_tokens)
    # generate_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generate_ids)
    # ]
    generate_ids = [
        output_ids[0:] for input_ids, output_ids in zip(inputs.input_ids, generate_ids)
    ]
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    # result = tokenizer.batch_decode(generate_ids,
    #                                 skip_special_tokens=True,
    #                                 clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    # if len(gen_output.shape())==2:
    #     for i in range(len(gen_output)):
    #         print()
    #         print(gen_output[i])
    #         print()
    # else:
    print(gen_output)

def prompt_eval(args, model_baseline, model_fintuned, model_fintunedC, tokenizer, tokenizerB, tokenizerC, device,
                prompts):
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)
        # inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print("+++++++++++++++++++++++++++++++")
        print("\n\n\n\n==========Baseline A: =========")
        r_base = generate(model_baseline,
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        print("==========finetune B: =========")
        text = tokenizerB.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizerB([text], return_tensors="pt").to(device)

        r_finetune_g = generate(model_fintuned,
                                tokenizerB,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_g)
        print("\n\n\n\n==========Baseline C: =========")
        text = tokenizerC.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizerC([text], return_tensors="pt").to(device)

        r_finetune_c = generate(model_fintunedC,
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_c)

        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        print("====================prompt end=============================")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


def main():
    args = parse_args()

    device = torch.device(get_accelerator().device_name(0))

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    model_baseline = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path_baseline,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline)

    # tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
    #                               fast_tokenizer=True,
    #                               # add_special_tokens=additional_special_tokens
    #                               )
    #
    # model_baseline = create_hf_model(AutoModelForCausalLM,
    #                                  args.model_name_or_path_baseline,
    #                                  tokenizer, None)
    #
    tokenizerB = load_hf_tokenizer(args.model_name_or_path_finetune,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, None)



    tokenizerC = load_hf_tokenizer(args.model_name_or_path_finetuneC,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    model_fintunedC = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetuneC,
                                     tokenizerC, None)

    model_baseline.to(device)
    model_fintuned.to(device)
    model_fintunedC.to(device)

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        prompts = [
            # "Please tell me about Microsoft in a few sentence?",
            # "Explain the moon landing to a 6 year old in a few sentences.",
            # "Write a short poem about a wise frog.",
            "Who was president of the United States in 1955?",
            "How does a telescope work?",
            "Why do birds migrate south for the winter?",
            "A ship traveling along a river has covered 24km\
             upstream and 28km downstream. For this journey, \
             it took half an hour less than for traveling 30km \
             upstream and 21km downstream, or half an hour more \
             than for traveling 15km upstream and 42km downstream, \
             assuming that both the ship and the river move uniformly.\
             Determine the speed of the ship in still water and the speed of the river."
            # "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhy do birds migrate south for the winter?<|im_end|>\n<|im_start|>assistant"

        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    prompt_eval(args, model_baseline, model_fintuned, model_fintunedC, tokenizer,tokenizerB,tokenizerC, device,
                prompts)


if __name__ == "__main__":
    main()
