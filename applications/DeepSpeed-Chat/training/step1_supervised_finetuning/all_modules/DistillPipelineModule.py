from deepspeed import PipelineModule

from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.runtime.pipe.module import TiedLayerSpec, LayerSpec
import deepspeed.runtime.utils as ds_utils
import torch
import torch.nn as nn
import copy
from functools import partial
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.utils import logger

class DistillPipelineModule(PipelineModule):

    def __init__(self,
                 teacher_layers,
                 student_layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seed_layers=False,
                 seed_fn=None,
                 base_seed=1234,
                 partition_method='parameters',
                 activation_checkpoint_interval=0,
                 activation_checkpoint_func=checkpointing.checkpoint,
                 checkpointable_layers=None,
                 dynamic_shape=False):

        super().__init__(layers = student_layers,
                         num_stages= num_stages,
                         topology = topology,
                         loss_fn = loss_fn,
                         seed_layers= seed_layers,
                         seed_fn = seed_fn,
                         base_seed = base_seed,
                         partition_method = partition_method,
                         activation_checkpoint_interval= activation_checkpoint_interval,
                         activation_checkpoint_func = activation_checkpoint_func,
                         checkpointable_layers = checkpointable_layers,
                         dynamic_shape= dynamic_shape)

        self._teacher_layer_specs = list(teacher_layers)
        self._teacher_num_layers = len(self._teacher_layer_specs)
        self._teacher_local_start = 0
        self._teacher_local_stop = None
        #will determine _teacher_local_start, _teacher_local_stop
        self._partition_teacher_layers(method=partition_method)

        self.forward_funcs_ref = []
        # self.forward_funcs_ref = []  #added for ref_model added by luke
        self._build_teacher()

        for layer in self.forward_funcs_ref:
            layer.to(get_accelerator().device_name(self.local_rank))

        #we move the whole subclass to device, not only for self.forward_funcs_ref
        self.to(get_accelerator().device_name(self.local_rank))

        #since teacher layer is not added to module list
        #so self.to() won't move teacher layer to device, but the module self is now cuda
        #then if input data is on cuda, there will be error
        #but if we do not move self.to(), then this module could be viewed on cpu
        #so it still works to receive data feeding in cpu, but very slowly


    def _build_teacher(self):
        specs = self._teacher_layer_specs
        for local_idx, layer in enumerate(specs[self._teacher_local_start:self._teacher_local_stop]):
            layer_idx = local_idx + self._teacher_local_start

            if isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                for param in module.parameters():
                    param.requires_grad = False
                self.forward_funcs_ref.append(module)
                # # start of create ref module for ref model  layer spec
                # parameter_names = [n for n, _ in module.named_parameters()]
                # for param_name in parameter_names:
                #     param = module.get_parameter(param_name)
                #     param.requires_grad = False
                # self.forward_funcs_ref.append(module)  # newly added for ref model
                # # end of create ref module

        # All pipeline parameters should be considered as model parallel in the context
        # of our FP16 optimizer
        for p in self.parameters():
            p.ds_pipe_replicated = False



    #copy from forward() function, but only for ref model feed forward
    def forward_ref_model(self, forward_input):

        def exec_range_func(start, end):
            ''' Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            '''

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs_ref[start:end]):
                    inputs = layer(inputs)
                return inputs

            return exec_func

        # if self.activation_checkpoint_interval == 0:
        func = exec_range_func(0, len(self.forward_funcs_ref))
        x = func(forward_input)
        # else:
        #     num_layers = len(self.forward_funcs_ref)
        #     x = forward_input
        #     for start_idx, is_checkpointable_result in \
        #         zip(range(0, num_layers, self.activation_checkpoint_interval), self.is_checkpointable_results):
        #
        #         end_idx = min(start_idx + self.activation_checkpoint_interval, num_layers)
        #
        #         funcs = self.forward_funcs_ref[start_idx:end_idx]
        #         # Since we either pass tensors or tuples of tensors without unpacking, we
        #         # need to be careful not to double-wrap tensors with tuple.
        #         if not isinstance(x, tuple):
        #             x = (x, )
        #
        #         if is_checkpointable_result:
        #             x = self.activation_checkpoint_func(exec_range_func(start_idx, end_idx), *x)
        #         else:
        #             x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _partition_teacher_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages with method {method}')

        method = method.lower()
        # Each stage gets a simple uniform number of layers.
        if method == 'parameters':
            param_counts = self._count_teacher_layer_params()
            #here we use different partition_balanced function
            self.teacher_parts = partition_balanced(weights=param_counts, num_parts=num_stages)

        #for debug, manually set
        self.teacher_parts = [0,2,3,6,10,19,31] # for distill 6 stages


        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.teacher_parts[stage]
                stop = self.teacher_parts[stage + 1]
                print(f'stage={stage} teacher_layers={stop - start}')
                for idx, layer in enumerate(self._teacher_layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    print(f'    {idx+start:2d}: {name}')
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')

        self._set_teacher_bounds(start=self.teacher_parts[stage_id], stop=self.teacher_parts[stage_id + 1])

    def _count_teacher_layer_params(self):
        param_counts = [0] * len(self._teacher_layer_specs)
        for idx, layer in enumerate(self._teacher_layer_specs):

            if isinstance(layer, LayerSpec):
                l = layer.build()
                # params = filter(lambda p: p.requires_grad, l.parameters())
                param_counts[idx] = sum(p.numel() for p in l.parameters())

        return param_counts

    def _set_teacher_bounds(self, start=None, stop=None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive and stop is
        exclusive. The default of None for both results in all layers being built
        locally.
        """
        # num_stages = self._topo.get_dim('pipe')
        # stage_id = self._topo.get_coord(self.global_rank).pipe

        self._teacher_local_start = start
        self._teacher_local_stop = stop


def partition_uniform(num_items, num_parts):
    import numpy
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = num_items // num_parts
    residual = num_items - (chunksize * num_parts)

    parts = numpy.arange(0, (num_parts + 1) * chunksize, chunksize)

    for i in range(residual):
        parts[i + 1:] += 1
    parts = parts.tolist()

    return parts


def partition_balanced(weights, num_parts):
    """
    use dynamic programming solve `The Linear Partition Problem`.
    see https://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
    """
    import numpy as np
    n = len(weights)
    m = num_parts

    if n <= m:
        return partition_uniform(n, m)

    dp_max = np.full((n + 1, m + 1), np.inf)
    dp_min = np.full((n + 1, m + 1), np.inf)
    dp_cost = np.full((n + 1, m + 1), np.inf)
    position = np.zeros((n + 1, m + 1), dtype=int)
    prefix_sum = np.zeros((n + 1))
    prefix_sum[1:] = np.cumsum(weights)

    dp_max[0, 0] = 0
    dp_cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, min(i, m) + 1):
            for k in range(i):
                max_sum = max(dp_max[k, j - 1], prefix_sum[i] - prefix_sum[k])
                min_sum = min(dp_min[k, j - 1], prefix_sum[i] - prefix_sum[k])
                cost = max_sum - min_sum
                if dp_cost[i, j] >= cost:
                    dp_cost[i, j] = cost
                    dp_max[i, j] = max_sum
                    dp_min[i, j] = min_sum
                    position[i, j] = k

    parts = [n]
    for i in reversed(range(1, m + 1)):
        parts.append(position[parts[-1], i])
    parts.reverse()

    return parts
