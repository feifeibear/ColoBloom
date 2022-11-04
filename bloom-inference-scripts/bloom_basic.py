import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch.distributed as dist
import bitsandbytes as bnb

from torch import Tensor, device, dtype, nn
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T", bound="torch.nn.Module")
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


    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m").to("cuda")

    # model inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
    print(logits)

def run_CAI():
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")

    kwargs = dict(
        device_map='balanced_low_0'
    )
    kwargs["load_in_8bit"] = True

    
    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m", **kwargs)
    # model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")

    
    print(model)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.cuda()

    # model inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits

    print(logits)

class Int8Params(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        requires_grad=False,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
    ):
        cls.has_fp16_weights = has_fp16_weights
        if data is None:
            data = torch.empty(0)
        cls.CB = None
        cls.SCB = None
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, data, has_fp16_weights=False, requires_grad=False):
        super(Int8Params, self).__init__
        B = data.contiguous().half().to(dist.get_rank())
        CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        self.CB = CB
        self.SCB = SCB
        self.data = CB
        self.has_fp16_weights = has_fp16_weights
        self.requires_grad = False


class Linear8bitLt(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        super(Linear8bitLt, self).__init__(
            input_features, output_features, bias
        )
        self.state = bnb.MatmulLtState()
        self.index = index

        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True
        # print(self.weight.data)
        self.weight = Int8Params(
            self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=False
        )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()
        
        # print(x.shape) # [6, 1024]
        # print(self.weight.shape) # [1536, 1024]
        # print(self.state.CB.shape) # [1536, 1024]
        # print(self.state.SCB.shape) # [1536]
        # print(self.bias.shape) # [1536]
        tmp_out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        tensor_list = [torch.zeros_like(tmp_out) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tmp_out)
        out = torch.cat(tensor_list, dim=1)

        if not self.state.has_fp16_weights:
            if not self.state.memory_efficient_backward and self.state.CB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
            elif self.state.memory_efficient_backward and self.state.CxB is not None:
                # For memory efficient backward, we convert 8-bit row major to turing/ampere format at each inference pass.
                # Thus, we delete CxB from the state. 
                del self.state.CxB

        return out

def run_tp():
    # init
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    # model
    # kwargs = dict(
    #     # device_map='balanced_low_0'
    # )
    # kwargs["load_in_8bit"] = True

    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m").to(rank)
    
    def replace_8bit_linear(model, threshold=6.0, modules_to_not_convert="lm_head"):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                replace_8bit_linear(module, threshold, modules_to_not_convert)

            if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                    # print("ori-weight:", model._modules[name].weight.data)
                    model._modules[name] = Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        threshold=threshold,
                    )
        return model
    
    model = replace_8bit_linear(model)

    for name, module in model.named_modules():
        if isinstance(module, Linear8bitLt):
            # print(module.weight.CB)
            # print(module.weight.SCB)
            weight_list = list(module.weight.data.chunk(world_size, dim=0))
            weight = weight_list[rank]
            module.weight = Int8Params(data=weight)
            # print(module.weight.CB)
            # print(module.weight.SCB)
            bias_list = list(module.bias.data.chunk(world_size, dim=0))
            bias = bias_list[rank]
            module.bias = nn.Parameter(data=bias)
            
        # elif isinstance(module, nn.Linear):
        #     weight_list = list(module.weight.chunk(world_size, dim=0))
        #     weight = weight_list[rank]
        #     module.weight = nn.Parameter(weight, requires_grad=False)

    # inputs
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)
    model = model
    for k, v in inputs.items():
        inputs[k] = v
        # dist.broadcast(inputs[k], src=0)
    
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits

    print(logits)

    # int8 + TP
def test():
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    
    module = Linear8bitLt(2, 4).to(rank)
    
    print(module.weight)
    weight_list = list(module.weight.data.chunk(world_size, dim=0))
    weight = weight_list[rank]
    module.weight = Int8Params(data=weight)
    print(module.weight.CB)
    
    x = torch.tensor([[ 0.3584,  1.0199], [-0.1108, -1.6668], [-0.1687,  1.8099], [-0.0600, -0.2149]], device=rank)

    out = module(x)
    print(out)

if __name__ == '__main__':
    run_tp()
