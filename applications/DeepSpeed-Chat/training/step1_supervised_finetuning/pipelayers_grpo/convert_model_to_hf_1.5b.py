import torch
from pathlib import Path
import os
from os.path import join
from shutil import copy
import argparse
import json
from transformers import set_seed, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import AutoModel
from safetensors.torch import save_file
from training.step1_supervised_finetuning.grpo import hash_tensor

def convert_model_to_hf(pipeline_model_dir, save_model_dir):
    model_static_dict = {}
    for path in Path(pipeline_model_dir).iterdir():
        print("已经处理文件：{}".format(path))
        if not path.name.startswith('layer'):
            continue
        small_static_dict = torch.load(path, map_location="cpu")
        layer_i = int(path.name.split('-')[0].replace('layer_', ''))
        if layer_i == 0:
            model_static_dict["model.embed_tokens.weight"] = small_static_dict["embed_tokens.weight"]
        elif layer_i == 30:
            model_static_dict["lm_head.weight"] = small_static_dict["embed_tokens.weight"]
        elif layer_i == 29: #norm layer
            model_static_dict['model.norm.weight'] = small_static_dict['norm.weight']

        elif layer_i <=28 and layer_i >= 1:
            for k, v in small_static_dict.items():
                model_static_dict["model." + k.replace("layer.", "layers.{}.".format(layer_i - 1), 1)] = v
    #

    # config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    # model = AutoModel.from_config(config)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    old_hash = hash_tensor(model.lm_head.weight.data)

    model.load_state_dict(model_static_dict)
    new_hash = hash_tensor(model.lm_head.weight.data)

    print(f"old hash:{old_hash}, new hash:{new_hash}")
    # state_dict = model.state_dict()
    # save_file(state_dict, join(save_model_dir,"model.safetensors"))
    model.save_pretrained(save_model_dir, safe_serialization=False)


    # torch.save(model_static_dict, join(save_model_dir, "pytorch_model.bin"))

# do not consider lora layer
#     model = convert_lora_to_linear_layer(model)

def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=fast_tokenizer)
    # tokenizer.pad_token = tokenizer.eos_token
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

    #no using dropout in inference
    # configure_dropout(model_config, dropout)

def set_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ori_model_dir', default='/ssd/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306', type=str, help='')
    parser.add_argument('--pipeline_model_dir', default='/ssd2/debug_20250723_sft/global_step1', type=str, help='')
    parser.add_argument('--save_model_dir', default='/ssd2/debug_20250723_sft', type=str, help='')
    return parser.parse_args()


if __name__ == '__main__':
    ages = set_args()
    convert_model_to_hf(ages.pipeline_model_dir, ages.save_model_dir)

    test_load_model(ages.save_model_dir)