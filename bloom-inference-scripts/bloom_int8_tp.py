import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch.distributed as dist
import os
from utils import replace_8bit_linear_tp, get_8bit_tp_model

# random_seed
torch.manual_seed(0)

def run_tp():
    # init
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
    model = model.half()
    # quantize
    model = replace_8bit_linear_tp(model).to(rank)

    # TP
    model = get_8bit_tp_model(model, rank, world_size)

    # inputs
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)

    # inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    
    print(logits)


def compare():
     # init
    world_size = 2
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = local_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
    model = model.half()
    model = replace_8bit_linear_tp(model).to(rank)

    # TP
    model = get_8bit_tp_model(model, rank, world_size)

    # inputs
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)

    # inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    output = outputs.logits

    # reference model
    model2 = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m", device_map='auto', load_in_8bit=True)
    outputs2 = model2(**inputs, labels=inputs["input_ids"])
    output2 = outputs2.logits

    assert torch.allclose(output, output2)==True, f'outputs from this method and hf method are not equal!'


if __name__ == '__main__':
    # run_tp()
    compare()
