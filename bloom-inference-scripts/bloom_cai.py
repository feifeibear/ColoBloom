import time
import os
import argparse
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig, AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from colossalai.tensor import ProcessGroup

import colossalai
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ShardSpec, ComputeSpec, ComputePattern, ColoParameter, ProcessGroup, ReplicaSpec

from_config = True
configuration = BloomConfig(hidden_size=1024,  # 64
                            n_layer=128,  # 2
                            n_head=32,  # 8
                            )
input_sentence = "Hello, my dog is cute"
max_new_tokens = 60

def print_rank0(str, rank = 0):
    if rank == 0:
        print(str)
    else:
        return
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_shard_int", required=False, type=bool, help="a flag inidicates init model in shards")
    parser.add_argument("--model_path", required=False, type=str, default="/data2/users/lczht/bloom-560m", help="used by dist launchers")
    parser.add_argument("--backend", required=False, type=str, default="colossalai", help = "backend of inference, [colossalai, torch, accelerate]")
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="float16")
    return parser.parse_args()


def run_torch(args):
    """
    run bloom inference using PyTorch
    """
    kwargs = dict()
    # kwargs = dict(
    #     device_map='balanced_low_0'
    # )
    # kwargs["load_in_8bit"] = True
    device = "cuda"
    model_path = args.model_path
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    model = BloomForCausalLM.from_pretrained(model_path, **kwargs).to(device)
    inputs = tokenizer(input_sentence, return_tensors="pt").to(device)

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # model inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits

def run_accelerate(args):
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = torch.cuda.device_count()
    print_rank0(f"Using {world_size} gpus", rank)
    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    model_name = args.model_path
    kwargs = dict(
        device_map="balanced",
    )
    infer_dtype = args.dtype
    if infer_dtype == "int8":
        print("Using `load_in_8bit=True` to use quanitized model")
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float16
    if not from_config:
        print("from pretrained")
        model = BloomForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        print("from config")
        # model1 = BloomForCausalLM(configuration)
        # model1.save_pretrained("temp_model_40B")
        model = BloomForCausalLM.from_pretrained("temp_model_40B", **kwargs)
    # for pn, param in model.named_parameters():
    #     print(param.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_tokens = tokenizer.batch_encode_plus([input_sentence], return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
            
    t_generate_span = 0
    print("inference start")
    for i in range(10):
        t_generate_start = time.time()
        outputs = model.generate(**input_tokens, **generate_kwargs)
        t_generate_span += time.time() - t_generate_start
    print_rank0(f"accelerate t_generate_span: {t_generate_span / 10}", rank)
    
def run_CAI(args):
    model_path = args.model_path
    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    colossalai.launch_from_torch(config={})
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pg = ProcessGroup(tp_degree=world_size)

    tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    if args.use_shard_int and from_config :
        print("from config, sharding")
        with ColoInitContext(device=torch.cuda.current_device(), dtype=torch.float16, default_pg=pg, default_dist_spec=ShardSpec([0], [pg.tp_world_size()])):
            model = BloomForCausalLM(configuration)
    else:
        with ColoInitContext(device=torch.cuda.current_device(), dtype=torch.float16, default_pg=pg):
            if from_config:
                print("from config")
                model = BloomForCausalLM(configuration)
            else:
                print("from pretrained")
                print(model_path)
                model = BloomForCausalLM.from_pretrained(model_path)

    # add ColossalAI Tensor Splitting Parallel
    def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
        spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
        if param.process_group.tp_world_size() == 1:
            param.set_process_group(pg)
        param.set_tensor_spec(*spec)
    def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
        split_param_single_dim_tp1d(0, param, pg)

    # shard_param_names = ['self_attention.dense.weight', 'dense_h_to_4h.weight', 'dense_4h_to_h.weight', 'self_attention.query_key_value.weight', 'word_embeddings.weight']
    shard_param_names = ['mlp', 'self_attention.dense', 'self_attention.query_key_value', 'word_embeddings.weight']
    num_params = 0
    num_params_unshard = 0
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=True):
            # reset process group for all parameters
            # param.set_process_group(pg)
            if hasattr(param, 'is_visited'):
                continue
            param_name = f"{mn}.{pn}"
            use_shard = False
            for keyword in shard_param_names:
                if keyword in param_name:
                    split_param_row_tp1d(param, pg)
                    # print('col slice', param_name)
                    use_shard = True
                    break
            # replicated param
            if not use_shard:
                param.set_dist_spec(ReplicaSpec())
            param.is_visited = True
            param.requires_grad_(False)
            num_params += param.numel()
            if use_shard:
                num_params_unshard += param.numel() * world_size
            else:
                num_params_unshard += param.numel()
    print('initialize TP OK')
    print(f"num_params: {num_params}")
    print(f"num_params_unshard: {num_params_unshard}")
    
    input_tokens = tokenizer(input_sentence, return_tensors="pt")
    for k, v in input_tokens.items():
        input_tokens[k] = v.cuda()
            
    print("inference start")
    # model inference
    t_generate_span = 0
    for i in range(10):
        t_generate_start = time.time()
        outputs = model.generate(**input_tokens, **generate_kwargs)
        # torch.cuda.synchronize()
        t_generate_span += time.time() - t_generate_start
    print_rank0(f"colossalai t_generate_span: {t_generate_span / 10}", rank)

if __name__ == '__main__':
    args = get_args()
    if args.backend == "colossalai":
        run_CAI(args)
    elif args.backend == "torch":
        run_torch(args)
    elif args.backend == "accelerate":
        run_accelerate(args)