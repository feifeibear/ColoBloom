import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch.distributed as dist
import bitsandbytes.functional as F
from torch import Tensor, device, dtype, nn

import os

class Int8Params(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        s=None,
        requires_grad=False,
    ):
        if data is None:
            data = torch.empty(0)
        cls.s = s
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, data, s=None, requires_grad=False):
        super(Int8Params, self).__init__
        self.s = s
        self.data = data


class Linear8bitLt(nn.Linear):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        index=None
    ):
        super(Linear8bitLt, self).__init__(
            input_features, output_features, bias
        )
        self.index = index
        self.weight = Int8Params(self.weight.data, requires_grad=False)
        #TP
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # quantize weight
        self.weight.data, self.weight.s = F.vectorwise_quant(self.weight.data, dim=0, quant_type="linear")
        self.weight.s = self.weight.s.to(self.rank)
        

    def forward(self, x):
        if self.bias is not None and self.bias.dtype != torch.float16:
            self.bias.data = self.bias.data.half()
        
        def gemm(A, B):
            A = A.to(torch.half)
            B = B.to(torch.half)
            return torch.matmul(A, B).to(torch.int8)

        def dequant(A, s1, s2, dtype):
            A = A.to(torch.float)
            A *= s1*s2 / 127**2
            return (A).to(dtype)
        
        qinput ,s_input= F.vectorwise_quant(x, dim=-1, quant_type="linear")
        qout = gemm(qinput, self.weight.data.t())
        out = dequant(qout, s_input, self.weight.s, x.dtype)

        tensor_list = [torch.zeros_like(out) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, out)
        out = torch.cat(tensor_list, dim=2)


        return out

def run_tp():
    # init
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
    
    def replace_8bit_linear(model, threshold=6.0, modules_to_not_convert="lm_head"):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                replace_8bit_linear(module, threshold, modules_to_not_convert)

            if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                    model._modules[name] = Linear8bitLt(
                        input_features=module.in_features,
                        output_features=module.out_features
                    )
        return model
    
    model = replace_8bit_linear(model).to(rank)

    for name, module in model.named_modules():
        if isinstance(module, Linear8bitLt):
            weight_list = list(module.weight.data.chunk(world_size, dim=0))
            weight = weight_list[rank]
            module.weight = Int8Params(data=weight, s=module.weight.s)

    # inputs
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)
    for k, v in inputs.items():
        inputs[k] = v
        # dist.broadcast(inputs[k], src=0)
    
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits

    print(logits)

if __name__ == '__main__':
    run_tp()
