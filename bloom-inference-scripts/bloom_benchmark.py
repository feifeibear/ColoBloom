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

INPUT_SENTENCE = "hello, my dog is cute"
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
    
    
    parser.add_argument("--use_config", dest="use_config", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=60)
    return parser.parse_args()


def run_torch(args):
    """
    run bloom inference using PyTorch
    """

    input_sentence = INPUT_SENTENCE
    
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
    from_config = True if args.use_config else False
    configuration = BloomConfig(hidden_size=args.hidden_size,  # 64
                                n_layer=args.n_layer,  # 2
                                n_head=args.n_head,  # 8
                                )
    input_sentence = INPUT_SENTENCE
    max_new_tokens = args.max_new_tokens
    
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = torch.cuda.device_count()
    print_rank0(f"Using {world_size} gpus", rank)
    
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
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        print("from config")
        filename = f"temp_model_{args.hidden_size}_{args.n_layer}_{args.n_head}"
        if not os.path.exists(filename):
            print(f"generate {filename}...") 
            model1 = BloomForCausalLM(configuration)
            model1.save_pretrained(filename)
            del model1
            print(f"generate {filename} done") 
        else:
            print(f"load {filename} done") 
            
        model = AutoModelForCausalLM.from_pretrained(filename, **kwargs)
    # for pn, param in model.named_parameters():
    #     print(param.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_tokens = tokenizer.batch_encode_plus([input_sentence], return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
            
    t_generate_span = 0
    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    for i in range(world_size):
        torch.cuda.reset_peak_memory_stats(i)
    # warmup
    for i in range(10):
        outputs = model.generate(**input_tokens, **generate_kwargs)
    print("inference start")
    
    for i in range(10):
        t_generate_start = time.time()
        outputs = model.generate(**input_tokens, **generate_kwargs)
        t_generate_span += time.time() - t_generate_start
    print_rank0(f"accelerate t_generate_span: {t_generate_span / 10}", rank)
    max_usage = 0
    for i in range(world_size):
        torch.cuda.reset_peak_memory_stats(i)
        memory = torch.cuda.max_memory_allocated(i)
        print(f"device {i} memory usage: {memory / 1024 /1024} MB")
        max_usage = max(max_usage, memory)
    print(f"max cuda memory usage: {max_usage / 1024 /1024} MB")
    
def run_CAI(args):
    from_config = True if args.use_config else False
    configuration = BloomConfig(hidden_size=args.hidden_size,  # 64
                                n_layer=args.n_layer,  # 2
                                n_head=args.n_head,  # 8
                                )
    input_sentence = INPUT_SENTENCE
    max_new_tokens = args.max_new_tokens
    
    model_path = args.model_path
    colossalai.launch_from_torch(config={})
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pg = ProcessGroup(tp_degree=world_size)
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    if args.use_shard_int and from_config :
        print("from config, sharded")
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
    max_usage = torch.cuda.max_memory_allocated(rank)
    print(f"max cuda memory usage: {max_usage / 1024 /1024} MB")
    print('initialize TP OK')
    print(f"num_params: {num_params}")
    print(f"num_params_unshard: {num_params_unshard}")
    
    input_tokens = tokenizer(input_sentence, return_tensors="pt")
    for k, v in input_tokens.items():
        input_tokens[k] = v.cuda()
    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.reset_peak_memory_stats(rank)
    # warmup
    for i in range(1):
        outputs = model.generate(**input_tokens, **generate_kwargs)
    print("inference start")
    # model inference
    t_generate_span = 0
    turn_num = 10
    for i in range(turn_num):
        t_generate_start = time.time()
        outputs = model.generate(**input_tokens, **generate_kwargs)
        # torch.cuda.synchronize()
        t_generate_span += time.time() - t_generate_start
    print_rank0(f"colossalai t_generate_span: {t_generate_span / turn_num}", rank)
    max_usage = torch.cuda.max_memory_allocated(rank)
    print(f"max cuda memory usage: {max_usage / 1024 /1024} MB")
    
def run_CAI_int8(args):
    from_config = True if args.use_config else False
    configuration = BloomConfig(hidden_size=args.hidden_size,  # 64
                                n_layer=args.n_layer,  # 2
                                n_head=args.n_head,  # 8
                                )
    input_sentence = INPUT_SENTENCE
    max_new_tokens = args.max_new_tokens
    
    from utils import replace_8bit_linear_tp, get_8bit_tp_model, get_8bit_tp_model_from_colomodule
    import torch.nn as nn
    
    model_path = args.model_path
    torch.manual_seed(0)
    colossalai.launch_from_torch(config={})
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pg = ProcessGroup(tp_degree=world_size)
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    if from_config:
        print("from config, sharded")
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
    # currently int8 mode is not integrated to colossalai. use isolated int8 tp
    # if from_config:
    #     model = BloomForCausalLM(configuration).half()
    # else :
    #     model = BloomForCausalLM.from_pretrained(model_path).half()
    model = get_8bit_tp_model_from_colomodule(model, world_size=world_size, rank=rank)
    # model = replace_8bit_linear_tp(model).to(rank)
    # model = get_8bit_tp_model(model, rank, world_size)
    
    # for pn, param in model.named_parameters(recurse=True):
    #     print(pn, param.dtype, param.device)
    num_params = 0
    for pn, param in model.named_parameters(recurse=True):
        if hasattr(param, 'is_visited'):
            continue
        num_params += param.numel()
        param.is_visited = True
    print("num_params: ", num_params)
    print('initialize INT8 TP OK')
    input_tokens = tokenizer(input_sentence, return_tensors="pt")
    for k, v in input_tokens.items():
        input_tokens[k] = v.cuda()
    
    generate_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.reset_peak_memory_stats(rank)
    # warmup
    for i in range(1):
        outputs = model.generate(**input_tokens, **generate_kwargs)
    # model inferences
    print("inference start")
    t_generate_span = 0
    turn_num = 10
    for i in range(turn_num):
        t_generate_start = time.time()
        outputs = model.generate(**input_tokens, **generate_kwargs)
        # torch.cuda.synchronize()
        t_generate_span += time.time() - t_generate_start
    print_rank0(f"colossalai t_generate_span: {t_generate_span / turn_num}", rank)
    max_usage = torch.cuda.max_memory_allocated(rank)
    print(f"max cuda memory usage: {max_usage / 1024 /1024} MB")
    
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.backend == "colossalai":
        if args.dtype == "float16":
            print("colossalai fp16")
            run_CAI(args)
        elif args.dtype == "int8":
            print("colossalai int8")
            run_CAI_int8(args)
    elif args.backend == "torch":
        run_torch(args)
    elif args.backend == "accelerate":
        print("accelerate")
        run_accelerate(args)
        
