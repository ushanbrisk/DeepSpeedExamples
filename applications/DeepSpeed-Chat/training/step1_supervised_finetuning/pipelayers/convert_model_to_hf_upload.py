import torch
from pathlib import Path
import os
from os.path import join
from shutil import copy
import argparse
import json
from transformers import set_seed, AutoTokenizer, AutoConfig, AutoModelForCausalLM


def convert_model_to_hf_qwen25_500m(pipeline_model_dir, save_model_dir):
    model_static_dict = {}
    for path in Path(pipeline_model_dir).iterdir():
        print("已经处理文件：{}".format(path))
        if not path.name.startswith('layer'):
            continue
        small_static_dict = torch.load(path, map_location="cpu")
        layer_i = int(path.name.split('-')[0].replace('layer_', ''))
        if layer_i == 0:
            model_static_dict["model.embed_tokens.weight"] = small_static_dict["embed_tokens.weight"]
        elif layer_i == 26:
            model_static_dict["lm_head.weight"] = small_static_dict["embed_tokens.weight"]
        elif layer_i == 25: #norm layer
            model_static_dict['model.norm.weight'] = small_static_dict['norm.weight']

        elif layer_i <=24 and layer_i >= 1:
            for k, v in small_static_dict.items():
                model_static_dict["model." + k.replace("layer.", "layers.{}.".format(layer_i - 1), 1)] = v
    #
    torch.save(model_static_dict, join(save_model_dir, "pytorch_model.bin"))
# do not consider lora layer
#     model = convert_lora_to_linear_layer(model)

def convert_model_to_hf_qwen25_3b(pipeline_model_dir, save_model_dir):
    model_static_dict = {}
    for path in Path(pipeline_model_dir).iterdir():
        print("已经处理文件：{}".format(path))
        if not path.name.startswith('layer'):
            continue
        small_static_dict = torch.load(path, map_location="cpu")
        layer_i = int(path.name.split('-')[0].replace('layer_', ''))
        if layer_i == 0:
            model_static_dict["model.embed_tokens.weight"] = small_static_dict["embed_tokens.weight"]
        elif layer_i == 38:
            model_static_dict["lm_head.weight"] = small_static_dict["embed_tokens.weight"]
        elif layer_i == 37: #norm layer
            model_static_dict['model.norm.weight'] = small_static_dict['norm.weight']

        elif layer_i <=36 and layer_i >= 1:
            for k, v in small_static_dict.items():
                model_static_dict["model." + k.replace("layer.", "layers.{}.".format(layer_i - 1), 1)] = v
    #
    torch.save(model_static_dict, join(save_model_dir, "pytorch_model.bin"))






def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=fast_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic, this is not implemented in openr1,
    # tokenizer.padding_side = 'right'
    return tokenizer

def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)

def test_load_model(model_path):
    model_json = os.path.join(model_path, "config.json")
    if os.path.exists(model_json):
        model_json_file = json.load(open(model_json))
        model_name = model_json_file.get("_name_or_path", model_path)
        tokenizer = get_tokenizer(model_name, fast_tokenizer=True)

    model_config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, torch_dtype=torch.bfloat16 )
    print(model)
    return model, tokenizer

    #no using dropout in inference
    # configure_dropout(model_config, dropout)

def set_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ori_model_dir', default='/ssd/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', type=str, help='')
    parser.add_argument('--pipeline_model_dir', default='/ssd2/debug_20250516/global_step20', type=str, help='')
    parser.add_argument('--save_model_dir', default='/ssd2/debug_20250516', type=str, help='')
    return parser.parse_args()

if __name__ == '__main__':
    ages = set_args()
    convert_model_to_hf_qwen25_500m(ages.pipeline_model_dir, ages.save_model_dir)

    model, tokenizer = test_load_model(ages.save_model_dir)
    # model = model.to("cuda:1")

    # test for connecting to vllm server
    # model = model.to(device)  #only for test, will use deepspeed.initialize() instead
    vllm_client = None

    from trl.extras.vllm_client import VLLMClient
    from vllm import SamplingParams

    #for test, comment it,
    vllm_client = VLLMClient(
        '0.0.0.0', 8000, connection_timeout=1200.0
    )
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

    test_updating = True
    if test_updating:
        # test for updating model parameter
        # For non-PEFT models, simply gather and update each parameter individually.
        for name, param in model.named_parameters():
            # print(f"name: {name}")
            vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        vllm_client.reset_prefix_cache()


#set cuda=1, model not to() any cuda, will be ok
#do not know why there will be errors in other cases