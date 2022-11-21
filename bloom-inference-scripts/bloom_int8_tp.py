import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import torch.distributed as dist
from utils import replace_8bit_linear_tp, get_8bit_tp_model
import colossalai

# torchrun --nproc_per_node=4 --master_port=1145 bloom_int8_tp.py 

def run_tp(world_size : int = 2):
    # init
    torch.manual_seed(0)
    colossalai.launch_from_torch(config={})
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"{rank} / {world_size}")
    model = BloomForCausalLM.from_pretrained("/data2/users/lczht/bloom-560m")
    model = model.half()
    # quantize
    model = replace_8bit_linear_tp(model).to(rank)
    # model = replace_8bit_linear_tp(model).to(rank)
    # TP
    model = get_8bit_tp_model(model, rank, world_size)
    # model.to(rank)
    for pn, param in model.named_parameters():
        print(pn, param.device)
    # model
    torch.cuda.reset_peak_memory_stats()
    # inputs
    tokenizer = BloomTokenizerFast.from_pretrained("/data2/users/lczht/bloom-560m")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)

    # inference
    
    outputs = model(**inputs, labels=inputs["input_ids"])
    logits = outputs.logits
    
    max_usage = torch.cuda.memory_allocated() # torch.cuda.max_memory_allocated()
    print(f"max cuda memory usage: {max_usage / 1024 /1024} MB")
    # print(logits)


def check_results():
     # init
    torch.manual_seed(0)
    colossalai.launch_from_torch(config={})
    world_size = dist.get_world_size()
    rank = dist.get_rank()

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
    run_tp()
    # check_results()
