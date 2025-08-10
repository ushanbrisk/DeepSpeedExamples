import copy
import time

from deepspeed.runtime.pipe.engine import PipelineEngine
from trl.data_utils import maybe_apply_chat_template, is_conversational, apply_chat_template
from trl.trainer.utils import pad
from trl.extras.profiling import profiling_context
import torch
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from collections.abc import Mapping
from torch import nn
import warnings
import os
from types import MethodType
from functools import reduce
from operator import mul
from deepspeed.runtime.pipe import schedule, p2p
from deepspeed.utils.timer import FORWARD_MICRO_TIMER, FORWARD_GLOBAL_TIMER, BACKWARD_MICRO_TIMER, \
    BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_INNER_GLOBAL_TIMER, \
    BACKWARD_REDUCE_MICRO_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, \
    STEP_MICRO_TIMER, STEP_GLOBAL_TIMER
from deepspeed import comm as dist
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from collections import OrderedDict

from . import grpo_schedule
from .grpo_schedule import InferenceSchedule
from . hash import hash_tensor
from . pipeline_data_loader import BatchDataBuffer

MEMORY_OPT_ALLREDUCE_SIZE = 500000000

BATCH_INPUT_TIMER = 'batch_input'
TRAIN_BATCH_TIMER = 'train_batch'
EVAL_BATCH_TIMER = 'eval_batch'
PIPE_SEND_OUTPUT_TIMER = 'pipe_send_output'
PIPE_SEND_GRAD_TIMER = 'pipe_send_grad'
PIPE_RECV_INPUT_TIMER = 'pipe_recv_input'
PIPE_RECV_GRAD_TIMER = 'pipe_recv_grad'

# The buffer size to store the meta data for each tensor.
TENSOR_META_SIZE = 256


class PipelineGRPOEngine(PipelineEngine):

    def __init__(self,
                 pipeline_grpo_config=None,
                 has_bool_tensors=False,
                 *super_args,
                 **super_kwargs):
        super().__init__(has_bool_tensors=has_bool_tensors, *super_args, **super_kwargs)
        #need tokenizer to do generation of answer, in sft, no need
        self.processing_class = pipeline_grpo_config['tokenizer']
        self.max_prompt_length = pipeline_grpo_config['max_prompt_length']
        self.use_vllm = pipeline_grpo_config['use_vllm']
        self.vllm_client = pipeline_grpo_config['vllm_client']
        self.num_generations = pipeline_grpo_config['num_generations']
        self.repetition_penalty = pipeline_grpo_config['repetition_penalty']
        self.temperature = pipeline_grpo_config['temperature']
        self.top_p = pipeline_grpo_config['top_p']
        self.top_k = pipeline_grpo_config['top_k']
        self.min_p = pipeline_grpo_config['min_p']
        self.max_completion_length = pipeline_grpo_config['max_completion_length']
        self.num_iterations = pipeline_grpo_config['num_iterations']
        self.grpo_beta = pipeline_grpo_config['grpo_beta']
        self.reward_funcs = pipeline_grpo_config['reward_funcs']
        self.reward_weights = pipeline_grpo_config['reward_weights']
        self.reward_processing_classes = [None] * len(self.reward_funcs)
        self.extended_buf_label = 0  #only useful for Inference mode, when need more buffer for 'label' to store reward data
        #since we use buffer for a total of gas steps, no need below function any more

        #self.set_batch_fn(self.prepare_inputs)  #should comment this func, when using buffer

        #need to send below value from main
        self.global_pipeline_step = 0  #need to consider when to add 1
        self._last_loaded_step = 0
        self._step = -1

        #for storing prompt_id, completiton_id, advantage
        #it will be prepared before entering pipeline
        self._buffered_inputs = [None] * self.micro_batches  #get from super class, equal gas

    #for evaluation purpose debug purpose
        self._eval_data_buffer = [None]

        #for storing ref_token_logps, will be saved after loss_fn()
        #should contains gas elements
        self._old_token_logps = [None] * self.micro_batches





        #reward communication channel
        self.first_reward_output_send = True #change to false after 1st time
        self.reward_recv_buf = None
        self._reward_buffer = []

        #data buffer for pipeline, having the same length
        self.collate_data_buffer = None

    #this function will read data for gas numbers, and make them of the same length
    def prepare_pipeline_total_inputs(self):
        batch = None

        if self.data_iterator is not None:
            batch = next(self.data_iterator)

        inputs = batch
        bool_is_conv = is_conversational(inputs[0])
        prompts = [x["prompt"] for x in inputs]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]


        # # First, have main process load weights if needed
        # if self.global_pipeline_step != self._last_loaded_step:
        #     self._move_model_to_vllm()  # after update param every gas steps, need to upload to server
        #     self._last_loaded_step = self.global_pipeline_step

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process

        # here not using ddp, so need to consider how the data format?
        # all_prompts_text = gather_object(prompts_text)
        all_prompts_text = prompts_text

        while True:
            # if self.accelerator.is_main_process:
            if self.global_rank == 0:
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                if True:
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        # n=min(self.num_generations, self.micro_batch_size),
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        # guided_decoding_regex=self.guided_decoding_regex, #do not know what it means
                    )
            else:  # other rank won't execute this function
                completion_ids = [None] * len(all_prompts_text)

            #print is to show the length of answer, to see whether it reach maximum limit
            print(f"answer shape:{[len(ele) for ele in completion_ids]}")



            # #for monitoring generated answer, for debug usage
            # sen_i = 0
            # for sentence in completion_ids:
            #     answer = self.processing_class.decode(sentence, skip_special_tokens=True)
            #     print(f"{sen_i}: {answer}")
            #     sen_i += 1
            #end of monitoring for debug usage


            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)

            #batch is of batch_size rows
            # if wait..:
            mean_rwd, std_rwd, advtg , rewards_per_func= self.compute_text_rewards(
                inputs = inputs,
                prompts = prompts,
                completion_ids = completion_ids,
                bool_is_conv = bool_is_conv)

            #here is to trick to jump out of reject sampling
            if advtg.sum() != 0.0 or advtg.sum() == 0.0:
            # if advtg.sum() != 0.0:
                break



        #get completion mask
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        send_to_next_stage = (prompt_completion_ids, attention_mask)

        logits_to_keep = completion_ids.size(1)
        send_to_last_stage = (prompt_completion_ids, torch.tensor([logits_to_keep]), advtg)

        #this print is for debugging evaluation info the same with train
        # print(f"prepare_pipeline_total_inputs(), advtg:{advtg}, capsulate data hash():{hash_tensor( prompt_completion_ids)}, shape:{prompt_completion_ids.shape}")

        return (send_to_next_stage, send_to_last_stage)

    #completion_ids, here we assume it is in cpu, not put in cuda
    def compute_text_rewards(self, inputs, prompts, completion_ids, bool_is_conv=False):
        #translate answers to texts
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if bool_is_conv:
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs))
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            # with profiling_context(self, reward_func_name):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                if bool_is_conv:
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )

                # reward_inputs = self._prepare_inputs_0(reward_inputs)
                with torch.inference_mode(): #what's the meaning?
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32)

        # if rewards_per_func.any(dim=-1)==True:
        if rewards_per_func.sum() != 0.0:
            print(f"reward_per_func: {rewards_per_func}")

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

#start of logging
        total_log = f"prompt:{prompts[0][1]['content']}\n\n\n\n"
        for i_ans in range(len(prompts)):
            one_ans = f"    idx:[{i_ans}]-------------------\n        completion: {completions[i_ans][0]['content']}\n        reward:{rewards_per_func[i_ans]}\n       answer:{inputs[i_ans]['answer']}\n"
            total_log = total_log + one_ans
        print(total_log)
#end of logging

        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if True:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        return (mean_grouped_rewards, std_grouped_rewards, advantages, rewards_per_func)



    #self._steps is increamented every next(), so in one train_step() , will
    #be incremented by gas
    #self.global_pipeline_step is incremented every gas next(), after one train_step, will
    #be increamented by 1
    def prepare_inputs(self, inputs):
        # mode = "eval" if self.control.should_evaluate else "train"
        mode = "train"
        if mode == "train":
            if self.global_pipeline_step % self.num_iterations == 0:
                #here we can choose between self._generate_and_score_completions_fake && self._generate_and_score_completions
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.micro_batches] = inputs
            else:#resue data
                inputs = self._buffered_inputs[self._step % self.micro_batches]
            self._step += 1
        # else:
        #     # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
        #     inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs
    ):
        device = self.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = self._prepare_inputs_0(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        # if self.args.use_vllm:
        if True:
            # First, have main process load weights if needed
            if self.global_pipeline_step != self._last_loaded_step:
                self._move_model_to_vllm()  #after update param every gas steps, need to upload to server
                #move upper function to last part of for loop outside in the main function
                #and use update param to vllm server
                #it will first read param from local disk files, merges them and upload
                self._last_loaded_step = self.global_pipeline_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process

            #here not using ddp, so need to consider how the data format?
            # all_prompts_text = gather_object(prompts_text)
            all_prompts_text = prompts_text

            # if self.accelerator.is_main_process:
            if self.global_rank == 0:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                #it still works under no ddp, if batch_size=4, num_gen=3, then it will select 2
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                #but wee need to constrain batch_size_per_gpu be multiple of num_generations
                #so that generated answer won't be wasted
                # with profiling_context(self, "vLLM.generate"):  #no need, it use wandb
                if True:
                    # import pickle   #it is fake generated, for test only
                    # with open('completion.pkl', 'rb') as file:
                    #     completion_ids = pickle.load(file)

                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        #guided_decoding_regex=self.guided_decoding_regex, #do not know what it means
                    )
            else: #other rank won't execute this function
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            # completion_ids = broadcast_object_list(completion_ids, from_process=0)
            # process_slice = slice(
            #     self.accelerator.process_index * len(prompts),
            #     (self.accelerator.process_index + 1) * len(prompts),
            # )
            # completion_ids = completion_ids[process_slice]

            #for test



            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # else:
        #     # Regular generation path
        #     with unwrap_model_for_generation(
        #         self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        #     ) as unwrapped_model:
        #         prompt_completion_ids = unwrapped_model.generate(
        #             prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
        #         )
        #
        #     # Compute prompt length and extract completion ids
        #     prompt_length = prompt_ids.size(1)
        #     prompt_ids = prompt_completion_ids[:, :prompt_length]
        #     completion_ids = prompt_completion_ids[:, prompt_length:]
        #
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # with torch.no_grad():
        #     # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
        #     # computation here, and use per_token_logps.detach() instead.
        #     #every gas steps, model will be updated
        #     #when num_iter>1, prompt
        #     # it will be used for when global_step%num_iterations > 0
        #     # but it still need to pass throught all pipeline stage
        #     # will feed to stage0
        #     if self.num_iterations > 1:
        #         old_per_token_logps = self._get_per_token_logps(
        #             self.model, prompt_completion_ids, attention_mask, logits_to_keep
        #         )
        #     else:
        #         old_per_token_logps = None
        #
        #     if self.beta == 0.0:
        #         ref_per_token_logps = None
        #     elif self.ref_model is not None:
        #         ref_per_token_logps = self._get_per_token_logps(
        #             self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
        #         )
        #     else:
        #         with self.accelerator.unwrap_model(self.model).disable_adapter():
        #             ref_per_token_logps = self._get_per_token_logps(
        #                 self.model, prompt_completion_ids, attention_mask, logits_to_keep
        #             )

        #so all data needed to feed stage0 is
        # prompt_completion_ids;
        # #attention_mask,
        # #logits_to_keep

        # Decode the generated completions, this part could be finished in stage0
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            # with profiling_context(self, reward_func_name):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = self._prepare_inputs_0(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if True:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # # Slice to keep only the local part of the data
        # process_slice = slice(
        #     self.accelerator.process_index * len(prompts),
        #     (self.accelerator.process_index + 1) * len(prompts),
        # )
        # advantages = advantages[process_slice]
        #
        # # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"
        #
        # if mode == "train":
        #     self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        # self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        #
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        # self._metrics[mode]["completion_length"].append(completion_length)
        #
        # # Calculate mean reward per function, but only for samples where the function was applied
        # for i, reward_func in enumerate(self.reward_funcs):
        #     if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
        #         reward_func_name = reward_func.config._name_or_path.split("/")[-1]
        #     else:
        #         reward_func_name = reward_func.__name__
        #     # Only calculate mean for samples where this reward function was applied (non-NaN values)
        #     mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        #     self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        # self._metrics[mode]["reward"].append(rewards.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        #
        # if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
        #     prompts_to_log = gather_object(prompts_text)
        #     completions_to_log = gather_object(completions_text)
        #     rewards_to_log = rewards.tolist()
        #
        #     if self.accelerator.is_main_process:
        #         if is_rich_available():
        #             print_prompt_completions_sample(
        #                 prompts_to_log,
        #                 completions_to_log,
        #                 rewards_to_log,
        #                 self.state.global_step,
        #             )
        #         if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
        #             import pandas as pd
        #
        #             # For logging
        #             table = {
        #                 "step": [str(self.state.global_step)] * len(rewards),
        #                 "prompt": prompts_to_log,
        #                 "completion": completions_to_log,
        #                 "reward": rewards.tolist(),
        #             }
        #             df = pd.DataFrame(table)
        #             wandb.log({"completions": wandb.Table(dataframe=df)})
        #
        # return {
        #     "prompt_ids": prompt_ids,
        #     "prompt_mask": prompt_mask,
        #     "completion_ids": completion_ids,
        #     "completion_mask": completion_mask,
        #     # "old_per_token_logps": old_per_token_logps,
        #     # "ref_per_token_logps": ref_per_token_logps,
        #     "advantages": advantages,
        # }

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        logits_to_keep = completion_ids.size(1)


        send_to_next_stage = (prompt_completion_ids,attention_mask)
        send_to_last_stage = (prompt_completion_ids, torch.tensor([logits_to_keep]), advantages)

        # fake_store = (prompt_completion_ids,attention_mask,logits_to_keep,advantages)
        # import pickle
        # file = open('fake.pickle', 'wb')
        # pickle.dump(fake_store, file)
        # file.close()
        return (send_to_next_stage, send_to_last_stage)

    def _generate_and_score_completions_fake(self, inputs):
        import pickle  # it is fake generated, for test only
        with open('fake.pickle', 'rb') as file:
            df  = pickle.load(file)
            prompt_completion_ids, attention_mask, logits_to_keep, advantages = df

        send_to_next_stage = (prompt_completion_ids, attention_mask)
        send_to_last_stage = (prompt_completion_ids, torch.tensor([logits_to_keep]), advantages)

        print(f"_generate_and_score_fake() , hash value[0->{self.num_stages-1}]: {hash_tensor(send_to_last_stage)}")

        return (send_to_next_stage, send_to_last_stage)



    def _exec_load_micro_batch(self, buffer_id):
        # print(f"_exec_load_micro_batch() called")
        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).start()
        if self.is_first_stage():
            batch = self._next_batch()
            loaded = None
            if torch.is_tensor(batch[0]):
                loaded = batch[0].clone().to(self.device).detach()
                if self._config.pipeline['activation_checkpoint_interval'] > 0 and self._config.pipeline[
                        'use_reentrant']:
                    loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], (tuple, list))
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    if self._config.pipeline['activation_checkpoint_interval'] > 0 and self._config.pipeline[
                            'use_reentrant']:
                        mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)
            self.pipe_buffers['inputs'][buffer_id] = loaded

        # if self.is_last_stage():
        #     loaded = batch[1]
        #     if torch.is_tensor(batch[1]):
        #         loaded = batch[1].to(self.device)
        #     # XXX: torch 1.6.0 DataLoader will auto convert tuple to list
        #     elif isinstance(batch[1], (tuple, list)):
        #         loaded = []
        #         for x in batch[1]:
        #             assert torch.is_tensor(x)
        #             x = x.to(self.device).detach()
        #             loaded.append(x)
        #         loaded = tuple(loaded)
        #     self.pipe_buffers['labels'][buffer_id] = loaded

            assert isinstance(batch[1], (tuple, list))
                # Assume list or tuple
            loaded = []
            for x in batch[1]:
                assert torch.is_tensor(x)
                mine = x.clone().detach().to(self.device)
                if self._config.pipeline['activation_checkpoint_interval'] > 0 and self._config.pipeline[
                            'use_reentrant']:
                    mine.requires_grad = mine.is_floating_point()
                loaded.append(mine)
            loaded = tuple(loaded)
            self.pipe_buffers['labels'][buffer_id] = loaded

            #newly added instruction
            # self._exec_send_rewards(buffer_id)


            # self._exec_recv_rewards(buffer_id)

        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).stop()

    def _exec_forward_pass(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('forward_luke').start()
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        #each stage need exectue forward_pass, so accumulated here to indicate sequence index
        self._step += 1

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])

            #for trim
            prompt_completion_ids,attention_mask = inputs
            original_length = attention_mask.shape[1]
            valid_length = max((attention_mask == 1).sum(dim=-1))
            prompt_completion_ids = prompt_completion_ids[:, :valid_length]
            attention_mask = attention_mask[:, :valid_length]
            inputs = (prompt_completion_ids, attention_mask)
            #end of trim
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            if self.pipe_partition_input_meta_cache is None:
                self.pipe_partition_input_meta_cache = inputs[0].to('cpu')
            part_input = PartitionedTensor.from_meta(meta=self.pipe_partition_input_meta_cache,
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][buffer_id] = inputs

        # inputs has no gradient because it is from a cloned tensor

        outputs = super(PipelineEngine,self).forward(inputs)
        print(f"step:{self.global_steps},stage:{self.stage_id}, buffer_id:{buffer_id}, inputs:{hash_tensor(inputs)}, outputs:{hash_tensor(outputs)}")


        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1, device=first_output.data.device)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        #for padding
        hidden_state, causal_mask = outputs
        expand_column = original_length  - valid_length
        additional = torch.zeros((hidden_state.shape[0], expand_column, hidden_state.shape[-1]), dtype=hidden_state.dtype,device=hidden_state.device).detach()
        hidden_state = torch.cat((hidden_state, additional),1)
        additional_mask = torch.zeros((hidden_state.shape[0], expand_column), dtype=causal_mask.dtype,device=causal_mask.device).detach()
        causal_mask = torch.cat((causal_mask, additional_mask),1)
        outputs = (hidden_state, causal_mask)
        #end of padding

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]


                #start of debug print, to verify evaluation data the same with train
                # prompt_completion_ids, logits_to_keep, advantages = labels
                # print(f"pid: {os.getpid()}, buffer_id:{buffer_id}, exec_forward_pass() get label data, adv:{advantages},data trim hash() : {hash_tensor(prompt_completion_ids)}, data trim shape:{prompt_completion_ids.shape}")
                #end of debug print

                self.loss = self.module.loss_fn(outputs,
                                                labels,
                                                self._old_token_logps,
                                                self.global_pipeline_step,
                                                self._step,
                                                )
            else:
                # Some models just return loss from forward()
                self.loss = outputs
            torch.cuda.empty_cache()  #newly add luke
            if self.eval_return_logits:
                self.outputs = outputs

            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

            def add_to_total_loss(_total_loss, _loss):
                if isinstance(_loss, torch.Tensor):
                    if _total_loss is None:
                        _total_loss = torch.zeros_like(_loss)
                    _total_loss += _loss.detach()
                else:
                    if _total_loss is None:
                        _total_loss = [torch.zeros_like(_l) for _l in _loss]
                    for _idx, _l in enumerate(_loss):
                        _total_loss[_idx] += _l.detach()
                return _total_loss

            self.total_loss = add_to_total_loss(self.total_loss, self.loss)

            # aggregate additional losses across gradient accumulation steps
            additional_losses = self.module.get_additional_losses()
            if additional_losses is not None:
                if self.total_additional_losses is None:
                    self.total_additional_losses = OrderedDict()
                for name, loss in additional_losses.items():
                    total = self.total_additional_losses[name] if name in self.total_additional_losses else None
                    self.total_additional_losses[name] = add_to_total_loss(total, loss)
        if self.wall_clock_breakdown():
            self.timers('forward_luke').stop()

        self.timers.log([
            'forward_luke',
            ])


    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].half()
            outputs = tuple(outputs)

        if self.dynamic_shape or self.first_output_send:
            self.first_output_send = False  #should send meta every time as size changes
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')
        print(f"_exec_send_activation() , ACT COMM[{self.stage_id}->{self.stage_id+1}:{buffer_id}] hash value: {hash_tensor(outputs)}")
        # Restore the boolean tensor
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].bool()
            outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).stop()

        self.timers.log([
            PIPE_SEND_OUTPUT_TIMER,
            ])

    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).start()

        recvd = None

        # Allocate the buffer if necessary
        if self.dynamic_shape or self.pipe_recv_buf is None:
        # if True:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                # XXX hardcode meta type
                if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                    buffer = self.meta_buffer

                p2p.recv(buffer, self.prev_stage)
                recvd[idx] = buffer.clone().detach()

            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            if self.has_attention_mask or self.has_bool_tensors:
                recvd[-1] = recvd[-1].bool()

            recvd = tuple(recvd)
            print(f"_exec_recv_activation() , ACT COMM[{self.stage_id-1}->{self.stage_id}:{buffer_id}] hash value: {hash_tensor(recvd)}")

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()
            # first element value specify which tensor need grad, only one now
            # print(f"rank: {self.global_rank}, stage:{self.stage_id}")

            # if len(recvd) == 10:    #the problem is caused by cos,sin, only consider 7, other case won't do anything
            #     for idx, buffer in enumerate(recvd):  #added by luke 2025.03.27 #change by luke 20250515
            #         if idx != int(recvd[0]):
            #             buffer.requires_grad = False


        self.pipe_buffers['inputs'][buffer_id] = recvd

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).stop()

        self.timers.log([
            PIPE_RECV_INPUT_TIMER,
            ])

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super(PipelineEngine,self).backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_MICRO_TIMER).start()
            self.timers(BACKWARD_GLOBAL_TIMER).start()
            self.timers(BACKWARD_INNER_MICRO_TIMER).start()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                if self.pipe_partition_output_meta_cache is None:
                    self.pipe_partition_output_meta_cache = outputs[0].to('cpu')
                part_output = PartitionedTensor.from_meta(meta=self.pipe_partition_output_meta_cache,
                                                          local_part=outputs[1],
                                                          group=self.grid.get_slice_parallel_group())
                self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
            else:
                # Already restored from partition
                self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            #print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
            if self.grad_partition_grad_layer_meta_cache is None:
                self.grad_partition_grad_layer_meta_cache = self.grad_layer[0].to('cpu')
            part_grad = PartitionedTensor.from_meta(meta=self.grad_partition_grad_layer_meta_cache,
                                                    local_part=self.grad_layer[1],
                                                    group=self.grid.get_slice_parallel_group())
            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            part_grad = None
            #print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point() and t.requires_grad]  # added luke 20250327
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            if not self._config.bfloat16_immediate_grad_update:
                self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).stop()
            self.timers(BACKWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_GLOBAL_TIMER).stop()

        self.mem_status('AFTER BWD')

        self.timers.log([
            BACKWARD_INNER_MICRO_TIMER,
            BACKWARD_INNER_GLOBAL_TIMER,
            BACKWARD_MICRO_TIMER,
            BACKWARD_GLOBAL_TIMER
            ])



    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_GRAD_TIMER).start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        if self.is_grad_partitioned:
            if isinstance(inputs, tuple):
                first_input = inputs[0]
                assert all([torch.is_tensor(elt) for elt in inputs[1:]])
                inputs_grad_tail = [elt.grad for elt in inputs[1:]]
            elif torch.is_tensor(inputs):
                first_input = inputs
                inputs_grad_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            assert torch.is_tensor(first_input)
            part = PartitionedTensor(tensor=first_input.grad, group=self.grid.get_slice_parallel_group())

            inputs = (part.to_meta(), part.data(), *inputs_grad_tail)

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.has_attention_mask or self.has_bool_tensors:
            inputs = list(inputs)
            inputs.pop()
            inputs = tuple(inputs)

        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            p2p.send(inputs.grad, self.prev_stage)
        else:
            # XXX terrible hacky branch
            if self.is_grad_partitioned:
                # First two sends are partitioned gradient
                p2p.send(inputs[0], self.prev_stage)
                p2p.send(inputs[1], self.prev_stage)
            else:
                for idx, buffer in enumerate(inputs):
                    # Skip tensors that will not produce a grad
                    if not buffer.is_floating_point():
                        assert buffer.grad is None
                        continue
                    # if len(inputs) == 10:  #only amend case when there is cos, sin
                    #     if idx != int(inputs[0]):       #added luke 20250327 #change by luke 20250515
                    #         continue
                    assert buffer.grad is not None
                    p2p.send(buffer.grad, self.prev_stage)
                    print(f"_exec_send_grad() idx:{idx}, GRAD COMM[{self.stage_id}->{self.stage_id-1}:{buffer_id}] hash value: {hash_tensor(buffer.grad)}")

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_GRAD_TIMER).stop()


    def _exec_recv_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_GRAD_TIMER).start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        if self.is_pipe_partitioned and not self.is_grad_partitioned:
            if self.pipe_partition_grad_meta_cache is None:
                self.pipe_partition_grad_meta_cache = outputs[0].to('cpu')
            part_output = PartitionedTensor.from_meta(meta=self.pipe_partition_grad_meta_cache,
                                                      local_part=outputs[1],
                                                      group=self.grid.get_slice_parallel_group())
            outputs[0].data = part_output.full()
            outputs = (outputs[0], *outputs[2:])
            # save for backward
            self.pipe_buffers['outputs'][buffer_id] = outputs

        # Allocate gradient if necessary
        if self.dynamic_shape or self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                self.grad_layer = self._allocate_or_extend_buffers(0, list(outputs.size()), outputs.dtype)
            else:
                # XXX This is a HACK
                # When we exchange activations/gradients, the two pipe stages
                # need to issue the send/recv with the same buffer sizes or
                # else there is a deadlock. The is_floating_point() filter is
                # used to avoid sending gradients for tensors that do not
                # produce gradients. When TP>1, we partition the first
                # activations/gradients across TP ranks to save communication
                # volume and memory. That partitioned tensor is represented as
                # two tensors: a 1/TPth chunk of the original data and also a
                # small LongTensor storing the metadata used to reconstruct on
                # the other side. When combined, the floating point filter also
                # filtered out the metadata tensor. This quick (hacky) fix just
                # branches on is_grad_partitioned so we don't filter out the
                # metadata tensor.
                if self.is_grad_partitioned:
                    sizes_and_dtypes = [(list(t.size()), t.dtype)
                                        for t in outputs[:2]] + [(list(t.size()), t.dtype)
                                                                 for t in outputs[2:] if t.is_floating_point()]
                else:
                    sizes_and_dtypes = [(list(t.size()), t.dtype) for t in outputs if
                                        t.is_floating_point() and t.requires_grad]  # added by luke 2025 03 27

                self.grad_layer = [
                    self._allocate_or_extend_buffers(i, size, dtype)
                    for i, (size, dtype) in enumerate(sizes_and_dtypes)
                ]

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                p2p.recv(buffer, self.next_stage)
                print(f"_exec_recv_grad() , idx:{idx}, GRAD COMM[{self.stage_id+1}->{self.stage_id}:{buffer_id}] hash value: {hash_tensor(buffer)}")

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_GRAD_TIMER).stop()


    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        print(f"entering _exec_reduce_tied_grads, stage id:{self.stage_id}")
        for weight, group in weight_group_list:
            print(
                f"exec_reduce_tied_grads before dist all_reduce, stage_id:{self.stage_id}, memory:{id(weight)}")
            grad = weight._hp_grad if self.using_bf16_optimizer else weight.grad
            if grad is not None:
                print(f"exec_reduce_tied_grads before dist all_reduce, stage_id:{self.stage_id}, size: {grad.shape}, value:{grad[0][0]}, sum:{grad.sum()}")
                dist.all_reduce(grad, group=group)

                print(f"exec_reduce_tied_grads after dist all_reduce, stage_id:{self.stage_id}, size: {grad.shape}, value:{grad[0][0]}, sum:{grad.sum()}")

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.using_bf16_optimizer:
                # PP+BF16 work for ZeRO Stage 1
                self._bf16_reduce_grads()
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers(STEP_MICRO_TIMER).start()
            self.timers(STEP_GLOBAL_TIMER).start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/lr', self.get_lr()[0], self.global_samples)]
            if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                self.summary_events.append(
                    (f'Train/Samples/loss_scale', self.optimizer.cur_scale, self.global_samples))
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers(STEP_MICRO_TIMER).stop()
            self.timers(STEP_GLOBAL_TIMER).stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    BATCH_INPUT_TIMER,
                    FORWARD_MICRO_TIMER,
                    BACKWARD_MICRO_TIMER,
                    BACKWARD_INNER_MICRO_TIMER,
                    BACKWARD_REDUCE_MICRO_TIMER,
                    STEP_MICRO_TIMER,
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    FORWARD_GLOBAL_TIMER,
                    BACKWARD_GLOBAL_TIMER,
                    BACKWARD_INNER_GLOBAL_TIMER,
                    BACKWARD_REDUCE_GLOBAL_TIMER,
                    STEP_GLOBAL_TIMER,
                ])

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        meta_buffer = torch.empty(TENSOR_META_SIZE, dtype=torch.int32, device=self.device)
        if isinstance(buffer, torch.Tensor):
            meta_buf_list = [
                0,  # type of data (0: tensor, 1: list (unused), 2: tuple)
                self.DTYPE_TO_ID[buffer.dtype],  # dtype
                len(buffer.size())  # ndims
            ]
            meta_buf_list.extend(buffer.size())
            assert len(
                meta_buf_list
            ) <= TENSOR_META_SIZE, f"Buffer for metadata is too small. Current buffer size: {TENSOR_META_SIZE} but required {len(meta_buf_list)}"
            meta_buffer[:len(meta_buf_list)].copy_(torch.tensor(meta_buf_list, dtype=torch.int32))
            p2p.send(meta_buffer, recv_stage)

        elif isinstance(buffer, tuple):
            meta_buf_list = [
                2,  # type of data (0: tensor, 1: list (unused), 2: tuple)
                len(buffer)  # num_tensors
            ]

            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                meta_buf_list.append(self.DTYPE_TO_ID[tensor.dtype])
                meta_buf_list.append(len(tensor.size()))
                meta_buf_list.extend(tensor.size())

            assert len(
                meta_buf_list
            ) <= TENSOR_META_SIZE, f"Buffer for metadata is too small. Current buffer size: {TENSOR_META_SIZE} but required {len(meta_buf_list)}"
            meta_buffer[:len(meta_buf_list)].copy_(torch.tensor(meta_buf_list, dtype=torch.int32))
            p2p.send(meta_buffer, recv_stage)

        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Returns:
            Allocated buffer for receiving from send_stage.
        """
        buffer = torch.empty(TENSOR_META_SIZE, dtype=torch.int32, device=self.device)
        p2p.recv(buffer, send_stage)

        recv_type = buffer[0].item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_dtype = self.ID_TO_DTYPE[buffer[1].item()]
            recv_ndims = buffer[2].item()
            recv_shape = buffer[3:3 + recv_ndims].tolist()
            return self._allocate_or_extend_buffers(0, recv_shape, recv_dtype)

        # List or tuple of tensors (recv_type == 1 (list) is currently unused)
        elif recv_type == 1 or recv_type == 2:
            num_tensors = buffer[1].item()

            buffers = []
            offset = 2
            for idx in range(num_tensors):
                recv_dtype = self.ID_TO_DTYPE[buffer[offset].item()]
                recv_ndims = buffer[offset + 1].item()
                recv_shape = buffer[offset + 2:offset + 2 + recv_ndims].tolist()
                offset += 2 + recv_ndims

                buffers.append(self._allocate_or_extend_buffers(idx, recv_shape, recv_dtype))

            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')


    def _exec_send_rewards(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('send_rewards').start()

        #we use label channel
        outputs = self.pipe_buffers['labels'][buffer_id]
        #outputs should contains all the  information needed to
        #transmit from stage0 to last stage

        # #start of print info, to verify label data integrity, in evaluation and train mode
        # prompt_completion_ids, logits_to_keep, advtg = outputs
        # print(
        #     f"pid: {os.getpid()}, buffer_id:{buffer_id},send_rewards(),send adv:{advtg}, data hash() : {hash_tensor(prompt_completion_ids)}, shape:{prompt_completion_ids.shape}")
        # #end of print info

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].half()
            outputs = tuple(outputs)

        if self.dynamic_shape or self.first_reward_output_send:  #luke 20250517
            self.first_reward_output_send = False
            self._send_reward_meta(outputs, self.num_stages - 1)  #here we need to transmit to last stage

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.self.num_stages - 1)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                p2p.send(buffer, self.num_stages - 1)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        # Restore the boolean tensor
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].bool()
            outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            self.timers('send_rewards').stop()

        self.timers.log([
            'send_rewards',
            ])

    def _exec_recv_rewards(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('receive rewards').start()

        recvd = None

        # Allocate the buffer if necessary
        if self.dynamic_shape or self.reward_recv_buf is None:
        # if True:
            self.reward_recv_buf = self._recv_reward_meta( 0 ) #here receive from first stage

        if isinstance(self.reward_recv_buf, torch.Tensor):
            p2p.recv(self.reward_recv_buf, 0)  #here receive from first stage
            recvd = self.reward_recv_buf.clone().detach()
            #recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.reward_recv_buf, tuple)
            recvd = [None] * len(self.reward_recv_buf)
            for idx, buffer in enumerate(self.reward_recv_buf):
                assert torch.is_tensor(buffer)
                # XXX hardcode meta type
                if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                    buffer = self.meta_buffer

                p2p.recv(buffer, 0)  #here receive from first stage
                recvd[idx] = buffer.clone().detach()

            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            if self.has_attention_mask or self.has_bool_tensors:
                recvd[-1] = recvd[-1].bool()

            recvd = tuple(recvd)

            print(f"_exec_recv_reward() , hash value[0->{self.stage_id}:{buffer_id}]: {hash_tensor(recvd)}")

            # for buffer in recvd:
            #     buffer.requires_grad = buffer.is_floating_point()

            # if len(recvd) == 10:    #the problem is caused by cos,sin, only consider 7, other case won't do anything
            #     for idx, buffer in enumerate(recvd):  #added by luke 2025.03.27 #change by luke 20250515
            #         if idx != int(recvd[0]):
            #             buffer.requires_grad = False


        self.pipe_buffers['labels'][buffer_id] = recvd  #here we should put it into labels channel


        #start of print info
        prompt_completion_ids, logits_to_keep, advtg = recvd
        print(
            f"pid: {os.getpid()}, recvd_rewards(),recv adv:{advtg}, data hash() : {hash_tensor(prompt_completion_ids)}, shape:{prompt_completion_ids.shape}")
        #end of print info



        if self.wall_clock_breakdown():
            self.timers('receive rewards').stop()

        self.timers.log([
            'receive rewards',
            ])

    #seems the same with _send_tensor_meta()
    def _send_reward_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        meta_buffer = torch.empty(TENSOR_META_SIZE, dtype=torch.int32, device=self.device)
        if isinstance(buffer, torch.Tensor):
            meta_buf_list = [
                0,  # type of data (0: tensor, 1: list (unused), 2: tuple)
                self.DTYPE_TO_ID[buffer.dtype],  # dtype
                len(buffer.size())  # ndims
            ]
            meta_buf_list.extend(buffer.size())
            assert len(
                meta_buf_list
            ) <= TENSOR_META_SIZE, f"Buffer for metadata is too small. Current buffer size: {TENSOR_META_SIZE} but required {len(meta_buf_list)}"
            meta_buffer[:len(meta_buf_list)].copy_(torch.tensor(meta_buf_list, dtype=torch.int32))
            p2p.send(meta_buffer, recv_stage)

        elif isinstance(buffer, tuple):
            meta_buf_list = [
                2,  # type of data (0: tensor, 1: list (unused), 2: tuple)
                len(buffer)  # num_tensors
            ]

            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                meta_buf_list.append(self.DTYPE_TO_ID[tensor.dtype])
                meta_buf_list.append(len(tensor.size()))
                meta_buf_list.extend(tensor.size())

            assert len(
                meta_buf_list
            ) <= TENSOR_META_SIZE, f"Buffer for metadata is too small. Current buffer size: {TENSOR_META_SIZE} but required {len(meta_buf_list)}"
            meta_buffer[:len(meta_buf_list)].copy_(torch.tensor(meta_buf_list, dtype=torch.int32))
            p2p.send(meta_buffer, recv_stage)

        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_reward_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Returns:
            Allocated buffer for receiving from send_stage.
        """
        buffer = torch.empty(TENSOR_META_SIZE, dtype=torch.int32, device=self.device)
        p2p.recv(buffer, send_stage)

        recv_type = buffer[0].item()

        # A single tensor will be sent.
        if recv_type == 0:
            recv_dtype = self.ID_TO_DTYPE[buffer[1].item()]
            recv_ndims = buffer[2].item()
            recv_shape = buffer[3:3 + recv_ndims].tolist()
            return self._allocate_or_extend_reward_buffers(0, recv_shape, recv_dtype)

        # List or tuple of tensors (recv_type == 1 (list) is currently unused)
        elif recv_type == 1 or recv_type == 2:
            num_tensors = buffer[1].item()

            buffers = []
            offset = 2
            for idx in range(num_tensors):
                recv_dtype = self.ID_TO_DTYPE[buffer[offset].item()]
                recv_ndims = buffer[offset + 1].item()
                recv_shape = buffer[offset + 2:offset + 2 + recv_ndims].tolist()
                offset += 2 + recv_ndims

                buffers.append(self._allocate_or_extend_reward_buffers(idx, recv_shape, recv_dtype))

            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')


    def _allocate_or_extend_reward_buffers(self, idx, shape, dtype):
        numel = reduce(mul, shape) if len(shape) > 0 else 1
        if len(self._reward_buffer) <= idx or self._reward_buffer[idx].numel() < numel:
            new_buf = self._allocate_buffer(shape, dtype=dtype, num_buffers=1)[0]
            if len(self._reward_buffer) <= idx:
                self._reward_buffer.append(new_buf)
            else:
                # del self._reward_buffer[idx]
                # torch.cuda.empty_cache()
                # gc.collect()
                self._reward_buffer[idx] = None
                self._reward_buffer[idx] = new_buf
            return self._reward_buffer[idx]
        else:
            return self._reward_buffer[idx].flatten()[:numel].view(shape)

    def _prepare_inputs_0(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        # if self.args.past_index >= 0 and self._past is not None:
        #     inputs["mems"] = self._past
        # if self.args.past_index >= 0 and self._past is not None:
        #     inputs["mems"] = self._past

        return inputs

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data


    def _move_model_to_vllm(self):
        pass


    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        grpo_schedule.OptimizerStep: _exec_optimizer_step,
        grpo_schedule.ReduceGrads: _exec_reduce_grads,
        grpo_schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        grpo_schedule.LoadMicroBatch: _exec_load_micro_batch,
        grpo_schedule.ForwardPass: _exec_forward_pass,
        grpo_schedule.BackwardPass: _exec_backward_pass,
        grpo_schedule.SendActivation: _exec_send_activations,
        grpo_schedule.RecvActivation: _exec_recv_activations,
        grpo_schedule.SendGrad: _exec_send_grads,
        grpo_schedule.RecvGrad: _exec_recv_grads,
        grpo_schedule.SendRwd: _exec_send_rewards,
        grpo_schedule.RecvRwd: _exec_recv_rewards
    }

    def train_batch(self, data_iter=None, tokenizer=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.

        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        #newly added
        self.reset_activation_shape()

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        if data_iter is not None:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.total_loss = None
        self.total_additional_losses = None
        self._compute_loss = True

        #for test of model weight hash value, to verify weight do updates
        layers_funcs = self.module.forward_funcs
        weight_hash = hash_tensor(list(layers_funcs[-1].named_parameters())[0][1].data)
        weight_test_sample = (list(layers_funcs[-1].named_parameters())[0][1].data[0][:10])
        print(f"name:{str(layers_funcs[-1].__class__).split('.')[-1]},module_name:{list(layers_funcs[-1].named_parameters())[0][0]},module_sum:{list(layers_funcs[-1].named_parameters())[0][1].data.sum()},layers_len:{len(layers_funcs)},stage:{self.stage_id}, step:{self.global_steps}, last layer weight hash:{weight_hash}, weight_test_sample:{weight_test_sample}")
        #end of test

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()




        #start of upload model to server
        #condition,   global_step%num_iterations==0 and step%gas==0
        #while in pipeline, only   global_step%==num_iterations==0

        if (self.global_pipeline_step % self.num_iterations == 0) and (self.global_pipeline_step != self._last_loaded_step):
            self._move_model_to_vllm()  #update model weight
            self._last_loaded_step = self.global_pipeline_step

        #end of uploading server
        #collect all layers from all cross processes

        #for test
        #will process gas data within one train_batch() loop
        #so we first prepare all gas datas before entering schedule commands loops
        #in old way, each exec_load_micro()will load and prepare one data
        if self.is_first_stage():
            if self.global_pipeline_step % self.num_iterations == 0:
                collate_data_buffer = BatchDataBuffer()  # should remove after one train_batch() call
                sample_count = 0
                for data_i in range(self.micro_batches):
                    batch = self.prepare_pipeline_total_inputs()
                    collate_data_buffer.add(batch)
                    sample_count += 1
                    #
                    # if mean_rwd.sum() > 0.0:
                    #     print(f"sample times: {sample_count}, rewards: \n{rewards}")
                    #     print(f"found non zero: { sample_count} times")
                    # else:   #all rewards are 0, useless, should resample
                    #     pass  #will implement later
                collate_data_buffer.adjust(padding_value=self.processing_class.pad_token_id)
                #store for later use, only when num_iterations>1, should store, otherwise, no need to
                #waste memory
                if self.num_iterations > 1:
                    self._buffered_inputs[0] = collate_data_buffer
            else:
                collate_data_buffer = self._buffered_inputs[0]

            #start of debug
            #for debug purpose, store data for evaluation use
            collate_data_buffer_eval_copy = BatchDataBuffer()
            collate_data_buffer_eval_copy.max_len = collate_data_buffer.max_len
            collate_data_buffer_eval_copy.data_num = collate_data_buffer.data_num
            if collate_data_buffer.buffer:
                for element in collate_data_buffer.buffer:
                    collate_data_buffer_eval_copy.buffer.append(copy.deepcopy(element))
            if collate_data_buffer.new_data_buffer:
                for element in collate_data_buffer.new_data_buffer:
                    collate_data_buffer_eval_copy.new_data_buffer.append(copy.deepcopy(element))

            #this is a brand new class data, copy from original, so not mixed with training data, prevent any disturbance
            self._eval_data_buffer[0] = collate_data_buffer_eval_copy
            #end of debug

            #here since contains all datas, so store in the first place
            self.collate_data_buffer = iter(collate_data_buffer)

        else:
            # while True:
            #     time.sleep(1)
            pass #for other processes, do nothing


        #end of test

        #entering scheduling cmds, will _exec_load_micro_data for gas times
        #originally call


        sched = grpo_schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)

        self.global_pipeline_step += 1   #added by luke

        if self.stage_id == 3:
            print(f"after executing all cmds of last stage")
        with torch.no_grad():
            self.agg_train_loss = self._aggregate_total_loss()

        self.timers(TRAIN_BATCH_TIMER).stop()

        if self.steps_per_print() is not None and self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                log_str = f'steps: {self.global_steps} loss: {self.agg_train_loss:0.4f} '
                if self.agg_additional_losses is not None:
                    for loss_name, loss_value in self.agg_additional_losses.items():
                        log_str += f'{loss_name}: {loss_value.item():0.4f} '
                log_str += f'iter time (s): {iter_time:0.3f} samples/sec: {tput:0.3f}'
                print(log_str)
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss', self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.steps_per_print() is not None and self.wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                PIPE_SEND_OUTPUT_TIMER,
                PIPE_SEND_GRAD_TIMER,
                PIPE_RECV_INPUT_TIMER,
                PIPE_RECV_GRAD_TIMER,
            ])

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.collate_data_buffer is not None:
            batch = next(self.collate_data_buffer)

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch


    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        super().reset_activation_shape()

        #reward communication channel
        self.first_reward_output_send = True #change to false after 1st time
        self.reward_recv_buf = None
        self._reward_buffer = []
        #data buffer for pipeline, having the same length
        self.collate_data_buffer = None
        self._eval_data_buffer = [None]  #also reset buffer for evaluation copy data

    def eval_batch(self,
                   reduce_output='avg',
                   tokenizer=None,
                   compute_loss=True,
                   return_logits=False,
                   bcast_loss=True,
                   num_micro_batches=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.

        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.module.eval()


        self.total_loss = None
        self.total_additional_losses = None
        self._compute_loss = True


        # Do the work
        self.timers(EVAL_BATCH_TIMER).start()

        #start of upload model to server
        #condition,   global_step%num_iterations==0 and step%gas==0
        #while in pipeline, only   global_step%==num_iterations==0

        #for test
        #will process gas data within one train_batch() loop
        #so we first prepare all gas datas before entering schedule commands loops
        #in old way, each exec_load_micro()will load and prepare one data
        if self.is_first_stage():

            #the common data is dataset, not data iterator
            collate_data_buffer = self._eval_data_buffer[0]
            #end of debug

            # #start of printing store data, just for debugging purpose
            # store_data_list = collate_data_buffer.new_data_buffer
            # for store_data in store_data_list:
            #     send_to_next_stage, send_to_last_stage = store_data
            #     prompt_completion_ids, logits_to_keep, advtg = send_to_last_stage
            #     prompt_completion_ids, attention_mask = send_to_next_stage
            #     data_fat_hash = hash_tensor(prompt_completion_ids)
            #     data_fat_shape = prompt_completion_ids.shape
            #     original_seq_len = prompt_completion_ids.shape[1]
            #     valid_length = max((attention_mask == 1).sum(dim=-1))
            #     prompt_completion_ids = prompt_completion_ids[:, :valid_length]
            #     print(
            #         f"pid: {os.getpid()}, eval_batch() fetch data store, receive adv:{advtg}, data fat hash:{data_fat_hash}, data fat shape:{data_fat_shape},data trim hash() : {hash_tensor(prompt_completion_ids)}, data trim shape:{prompt_completion_ids.shape}")
            # #end of printing store data

            #here since contains all datas, so store in the first place
            # temp_data_buffer = self.collate_data_buffer
            self.collate_data_buffer = iter(collate_data_buffer)

        else:
            # while True:
            #     time.sleep(1)
            pass #for other processes, do nothing


        #end of test

        #entering scheduling cmds, will _exec_load_micro_data for gas times
        #originally call

        # set the number micro batches in case the user chose value than training
        micro_batches = self.micro_batches if num_micro_batches is None else num_micro_batches
        self._compute_loss = compute_loss
        eval_output = None

        sched = grpo_schedule.InferenceSchedule(micro_batches=micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        # self.global_pipeline_step += 1   #added by luke

        self.timers(EVAL_BATCH_TIMER).stop()

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output, micro_batches=micro_batches)

        if compute_loss and (bcast_loss or self.monitor.enabled):
            eval_output = self._bcast_pipe_scalar(eval_output)

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', eval_output.mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.eval_return_logits = False
        if return_logits:
            outputs = self.outputs
            self.outputs = None
            return eval_output, outputs
        return eval_output
