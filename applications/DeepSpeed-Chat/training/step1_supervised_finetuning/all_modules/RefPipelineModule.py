from deepspeed import PipelineModule

from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.runtime.pipe.module import TiedLayerSpec, LayerSpec
import deepspeed.runtime.utils as ds_utils
import torch
import torch.nn as nn
import copy
from functools import partial
from deepspeed.accelerator import get_accelerator

class RefPipelineModule(PipelineModule):

    def __init__(self,
                 layers,
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
        self.forward_funcs_ref = []  # added for ref_model added by luke
        super().__init__(layers = layers,
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


        for layer in self.forward_funcs_ref:
            layer.to(get_accelerator().device_name(self.local_rank))

    def _build(self):
        specs = self._layer_specs

        for local_idx, layer in enumerate(specs[self._local_start:self._local_stop]):
            layer_idx = local_idx + self._local_start
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    ds_utils.set_random_seed(self.base_seed + layer_idx)

            # Recursively build PipelineModule objects
            if isinstance(layer, PipelineModule):
                raise NotImplementedError('RECURSIVE BUILD NOT YET IMPLEMENTED')

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, nn.Module):
                name = str(layer_idx)
                self.forward_funcs.append(layer)
                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, layer)

                # start of create ref module for ref model nn.Module
                module_ref = copy.deepcopy(layer)
                parameter_names = [n for n, _ in module_ref.named_parameters()]
                for param_name in parameter_names:
                    param = module_ref.get_parameter(param_name)
                    param.requires_grad = False
                self.forward_funcs_ref.append(module_ref)  # newly added for ref model

                # end of create ref module

            # TiedLayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, TiedLayerSpec):
                # Build and register the module if we haven't seen it before.
                if layer.key not in self.tied_modules:
                    self.tied_modules[layer.key] = layer.build()
                    self.tied_weight_attrs[layer.key] = layer.tied_weight_attr

                if layer.forward_fn is None:
                    # Just use forward()
                    self.forward_funcs.append(self.tied_modules[layer.key])

                    # start of create ref module for ref model TiedLayerSpec
                    module_ref = copy.deepcopy(self.tied_modules[layer.key])
                    # if self.stage_id == self.num_stages - 1:
                    #     operation = getattr(module_ref, "set_idle_ref_model", None)
                    #     if callable(operation):
                    #         operation()

                    parameter_names = [n for n, _ in module_ref.named_parameters()]
                    for param_name in parameter_names:
                        param = module_ref.get_parameter(param_name)
                        param.requires_grad = False
                    self.forward_funcs_ref.append(module_ref)  # newly added for ref model

                    # end of create ref module
                else:
                    # User specified fn with args (module, input)
                    self.forward_funcs.append(partial(layer.forward_fn, self.tied_modules[layer.key]))

                    # start of create ref module for ref model TiedLayerSpec loss fn
                    module_ref = copy.deepcopy(self.tied_modules[layer.key])
                    parameter_names = [n for n, _ in module_ref.named_parameters()]
                    for param_name in parameter_names:
                        param = module_ref.get_parameter(param_name)
                        param.requires_grad = False
                    self.forward_funcs_ref.append(partial(layer.forward_fn, module_ref))  # newly added for ref model
                    # end of create ref module

            # LayerSpec objects contain an nn.Module that should be allocated now.
            elif isinstance(layer, LayerSpec):
                module = layer.build()
                name = str(layer_idx)
                self.forward_funcs.append(module)

                self.fwd_map.update({name: len(self.forward_funcs) - 1})
                self.add_module(name, module)

                # start of create ref module for ref model  layer spec
                module_ref = copy.deepcopy(module)
                parameter_names = [n for n, _ in module_ref.named_parameters()]
                for param_name in parameter_names:
                    param = module_ref.get_parameter(param_name)
                    param.requires_grad = False
                self.forward_funcs_ref.append(module_ref)  # newly added for ref model
                # end of create ref module


            # Last option: layer may be a functional (e.g., lambda). We do nothing in
            # that case and just use it in forward()
            else:
                self.forward_funcs.append(layer)
                self.forward_funcs_ref.append(copy.deepcopy(layer))  # just copy for ref model, no use , maybe wrong

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
