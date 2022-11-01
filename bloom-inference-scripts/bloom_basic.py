import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch.distributed as dist
from colossalai.tensor import ProcessGroup

import colossalai
from colossalai.tensor import ShardSpec, ComputeSpec, ComputePattern, ColoInt8Parameter, ProcessGroup, ColoTensor, ColoParameter

from torch import nn
from typing import Iterator, Tuple, Union
from bitsandbytes.nn.modules import Linear8bitLt


def _named_params_with_replica(
    module: nn.Module,
    prefix: str = '',
    recurse: bool = True,
) -> Iterator[Tuple[str, Union[nn.Parameter, ColoTensor]]]:
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    for mod_prefix, mod in modules:
        for name, val in mod._parameters.items():
            if val is None:
                continue
            name = mod_prefix + ('.' if mod_prefix else '') + name
            yield name, val

def ColoModulize(module):
    """
    Replacing the parameters() and named_parameters() with our customized ones
    """

    module._colo_visited = True

def to_colo_tensor(module):
    name_list = []
    for name, param in _named_params_with_replica(module):
        if isinstance(param, ColoTensor):
            continue

        split = name.rfind('.')
        if split >= 0:    # param in submodule
            module_name = name[:split]
            param_name = name[split + 1:]
        else:
            module_name = ''    # param in current module
            param_name = name
        name_list.append((module_name, param_name))   
    
    replaced_tensors = dict(
    )    # record mapping between (torch.Tensor, ColoTensor) to distinguish the same reference
    for module_name, param_name in name_list:
        submodule = module.get_submodule(module_name)
        if isinstance(submodule, Linear8bitLt):
            param = submodule.get_parameter(param_name)
            # print(param)
            if param in replaced_tensors:
                colo_param = replaced_tensors[param]
            else:
                colo_param = ColoInt8Parameter(param, requires_grad=False)
                # add mapping record
                replaced_tensors[param] = colo_param
        else:
            param = submodule.get_parameter(param_name)
            if param in replaced_tensors:
                colo_param = replaced_tensors[param]
            else:
                colo_param = ColoParameter(param, requires_grad=False)
                # add mapping record
                replaced_tensors[param] = colo_param
        delattr(submodule, param_name)
        setattr(submodule, param_name, colo_param)
        colo_param.shared_param_modules.append(submodule)

    ColoModulize(module)
    return module



def run_torch():
    kwargs = dict(
        device_map='balanced_low_0'
    )
    kwargs["load_in_8bit"] = True
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m", **kwargs)
    # model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m").cuda()
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.to("cuda")

    # model inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
    print(logits)



def run_CAI():
    colossalai.launch_from_torch(config={})

    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")

    kwargs = dict(
        device_map='balanced_low_0'
    )
    kwargs["load_in_8bit"] = True

    
    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m", **kwargs)
    # model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")

    model = to_colo_tensor(model)

    print(model)
    # print(model)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.cuda()

    # for name, p in model.named_parameters():
    #     print(name)

    # add ColossalAI Tensor Splitting Parallel
    def split_param_single_dim_tp1d(dim: int, param: ColoInt8Parameter, pg: ProcessGroup):
        spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
        if param.process_group.tp_world_size() == 1:
            param.set_process_group(pg)
        param.set_tensor_spec(*spec)

    def split_param_row_tp1d(param: ColoInt8Parameter, pg: ProcessGroup):
        split_param_single_dim_tp1d(0, param, pg)

    def split_param_col_tp1d(param: ColoInt8Parameter, pg: ProcessGroup):
        split_param_single_dim_tp1d(-1, param, pg)

    pg = ProcessGroup(tp_degree=dist.get_world_size())
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # reset process group for all parameters
            param.set_process_group(pg)

            if 'dense_h_to_4h.weight' in pn or 'self_attention.query_key_value' in pn or 'mlp.dense_4h_to_h' in pn:
                split_param_row_tp1d(param, pg)  # colmn slice 


    # model inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits

    print(logits)

if __name__ == '__main__':
    run_CAI()
