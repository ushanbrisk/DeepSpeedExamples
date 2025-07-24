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

class PipelineSFTRefModelEngine(PipelineEngine):

    def __init__(self,
                 has_bool_tensors=False,
                 use_ref_model=False,
                 *super_args,
                 **super_kwargs):
        super().__init__(has_bool_tensors=has_bool_tensors,
                         *super_args,
                         **super_kwargs)
        self.use_ref_model = use_ref_model  #newly added for ref model

        if self.use_ref_model:
            self.pipe_buffers = {
                'inputs': [],  # batch input and received activations
                'labels': [],  # labels from batch input
                'outputs': [],  # activations
                'output_tensors': [],  # tensor object to preserve backward graph
                'inputs_ref':[],        #added for ref model
                'outputs_ref':[]        #added for ref model
            }


        self.pipe_recv_buf_ref_model = None #used for ref model


    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        super().reset_activation_shape()
        if self.use_ref_model:
            self.pipe_recv_buf_ref_model = None

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

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)

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

    def eval_batch(self,
                   data_iter,
                   return_logits=False,
                   compute_loss=True,
                   reduce_output='avg',
                   bcast_loss=True,
                   num_micro_batches=None):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.eval_return_logits = return_logits
        self.module.eval()

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

        eval_output = None

        self._compute_loss = compute_loss

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # set the number micro batches in case the user chose value than training
        micro_batches = self.micro_batches if num_micro_batches is None else num_micro_batches

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=micro_batches, stages=self.num_stages, stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(self.fwd_outputs, reduce=reduce_output, micro_batches=micro_batches)

        if compute_loss and (bcast_loss or self.monitor.enabled):
            eval_output = self._bcast_pipe_scalar(eval_output)

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', eval_output.mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()
        self.eval_return_logits = False
        if return_logits:
            outputs = self.outputs
            self.outputs = None
            return eval_output, outputs
        return eval_output


    def _exec_schedule(self, pipe_schedule):
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                # if self.stage_id >= 0:
                #     print(f"stage_id: {self.stage_id}, step_cmds:{ str(step_cmds)}, cmd: {str(cmd)}")
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self._exec_instr(**cmd.kwargs)


    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)

        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
            if self.use_ref_model:
                inputs_ref = tuple(t.clone() for t in self.pipe_buffers['inputs_ref'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()
            if self.use_ref_model:
                inputs_ref = self.pipe_buffers['inputs_ref'][buffer_id].clone()

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
        outputs = super(PipelineEngine, self).forward(inputs)
        if self.use_ref_model:
            outputs_ref = self.module.forward_ref_model(inputs_ref)

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

        self.pipe_buffers['outputs'][buffer_id] = outputs
        if self.use_ref_model:
            self.pipe_buffers['outputs_ref'][buffer_id] = outputs_ref #not necessary

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                if self.use_ref_model:
                    # ref_hidden = self.pipe_buffers['outputs_ref'][buffer_id]
                    # logps = outputs_ref
                    self.loss = self.module.loss_fn(outputs, labels, outputs_ref)
                else:
                    self.loss = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs
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




    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).start()

        batch = self._next_batch()

        if self.is_first_stage():
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
                # added by luke 2025 03 27  maybe wrong,  idx!= loaded[3], it may be wrong
                if len(loaded) == 7:
                    for idx, buffer in enumerate(loaded):
                        if idx != loaded[3]:
                            buffer.requires_grad=False
            self.pipe_buffers['inputs'][buffer_id] = loaded

            if self.use_ref_model:
                #here use the same variable, but do not need it in label buffer since label is shared between ref model and updated model
                self.pipe_buffers['inputs_ref'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            # XXX: torch 1.6.0 DataLoader will auto convert tuple to list
            elif isinstance(batch[1], (tuple, list)):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).stop()



    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        if self.use_ref_model:
            outputs_ref = self.pipe_buffers['outputs_ref'][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].half()
            outputs = tuple(outputs)
            if self.use_ref_model:
                outputs_ref = list(outputs_ref)
                outputs_ref[-1] = outputs_ref[-1].half()
                outputs_ref = tuple(outputs_ref)

        if self.dynamic_shape or self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        #for ref model useage
        if self.use_ref_model:
            if isinstance(outputs_ref, torch.Tensor):
                p2p.send(outputs_ref, self.next_stage)
            elif isinstance(outputs_ref, tuple):
                for idx, buffer in enumerate(outputs_ref):
                    p2p.send(buffer, self.next_stage)
            else:
                raise NotImplementedError('Could not send output of type '
                                          f'{type(outputs_ref)}')




        # Restore the boolean tensor
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].bool()
            outputs = tuple(outputs)
            if self.use_ref_model:
                outputs_ref = list(outputs_ref)
                outputs_ref[-1] = outputs_ref[-1].bool()
                outputs_ref = tuple(outputs_ref)

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).stop()


    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).start()

        recvd = None

        # Allocate the buffer if necessary
        if self.dynamic_shape or self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
            if self.use_ref_model:
                self.pipe_recv_buf_ref_model = copy.deepcopy(self.pipe_recv_buf)


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

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()
            # first element value specify which tensor need grad, only one now
            # print(f"rank: {self.global_rank}, stage:{self.stage_id}")

            if len(recvd) == 7:    #the problem is caused by cos,sin, only consider 7, other case won't do anything
                for idx, buffer in enumerate(recvd):  #added by luke 2025.03.27
                    if idx != int(recvd[0]):
                        buffer.requires_grad = False
        self.pipe_buffers['inputs'][buffer_id] = recvd

#############################################################
#############################################################
        ###for usage of refer model
        if self.use_ref_model:
            if isinstance(self.pipe_recv_buf_ref_model, torch.Tensor):
                p2p.recv(self.pipe_recv_buf_ref_model, self.prev_stage)
                recvd_ref_model = self.pipe_recv_buf_ref_model.clone().detach()
                recvd_ref_model.requires_grad = recvd_ref_model.is_floating_point()
            else:
                assert isinstance(self.pipe_recv_buf_ref_model, tuple)
                recvd_ref_model = [None] * len(self.pipe_recv_buf_ref_model)
                for idx, buffer in enumerate(self.pipe_recv_buf_ref_model):
                    assert torch.is_tensor(buffer)

                    p2p.recv(buffer, self.prev_stage)
                    recvd_ref_model[idx] = buffer.clone().detach()

                # NCCL does not like to send torch.BoolTensor types, so un-cast the
                # attention mask
                if self.has_attention_mask or self.has_bool_tensors:
                    recvd_ref_model[-1] = recvd_ref_model[-1].bool()

                recvd_ref_model = tuple(recvd_ref_model)

                for buffer in recvd_ref_model:
                    buffer.requires_grad = False
                # first element value specify which tensor need grad, only one now
                # print(f"rank: {self.global_rank}, stage:{self.stage_id}")



            self.pipe_buffers['inputs_ref'][buffer_id] = recvd_ref_model
########## end of ref model recv activation ###########

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).stop()

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

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.using_bf16_optimizer:
                # PP+BF16 work for ZeRO Stage 1
                self._bf16_reduce_grads()
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

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
        for weight, group in weight_group_list:
            grad = weight._hp_grad if self.using_bf16_optimizer else weight.grad
            if grad is not None:
                dist.all_reduce(grad, group=group)

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super(PipelineEngine, self).backward(self.loss)
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
                    if len(inputs) == 7:  #only amend case when there is cos, sin
                        if idx != int(inputs[0]):       #added luke 20250327
                            continue
                    assert buffer.grad is not None
                    p2p.send(buffer.grad, self.prev_stage)

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

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_GRAD_TIMER).stop()




    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

