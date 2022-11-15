import argparse
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig
import torch.distributed as dist
from colossalai.tensor import ProcessGroup

import colossalai
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor import ShardSpec, ComputeSpec, ComputePattern, ColoParameter, ProcessGroup, ReplicaSpec

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_shard_int", required=False, type=bool, help="a flag inidicates init model in shards")
    parser.add_argument("--model_path", required=False, type=str, default="/data2/users/lczht/bloom-560m", help="used by dist launchers")
    parser.add_argument("--backend", required=False, type=str, default="colossalai", help = "backend of inference, [colossalai, torch]")
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

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(device)

    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # model inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits


def run_CAI(args):
    model_path = args.model_path

    colossalai.launch_from_torch(config={})
    world_size = dist.get_world_size()
    pg = ProcessGroup(tp_degree=world_size)

    tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    cfg = BloomConfig()
    if args.use_shard_int:
        dist_spec = ShardSpec([-1], [world_size])
        with ColoInitContext(device=torch.device('cuda'), default_pg=pg, default_dist_spec=dist_spec):
            model = BloomForCausalLM(cfg)
    else:
        with ColoInitContext(device=torch.device('cuda')):
            model = BloomForCausalLM(cfg)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    for k, v in inputs.items():
        inputs[k] = v.cuda()

    # add ColossalAI Tensor Splitting Parallel
    def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
        spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
        param.set_tensor_spec(*spec)

    def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
        split_param_single_dim_tp1d(0, param, pg)

    def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
        split_param_single_dim_tp1d(-1, param, pg)

    shard_param_names = ['self_attention.dense.weight', 'dense_h_to_4h.weight', 'dense_4h_to_h.weight', 'self_attention.query_key_value.weight', 'word_embeddings.weight']
    for mn, module in model.named_modules():
        for pn, param in module.named_parameters(recurse=False):
            # reset process group for all parameters
            # param.set_process_group(pg)
            if hasattr(param, 'is_visited'):
                continue
            param_name = f"{mn}.{pn}"
            
            use_shard = False
            for keyword in shard_param_names:
                if keyword in param_name:
                    split_param_col_tp1d(param, pg)  # colmn slice 
                    print('col slice', param_name)
                    use_shard = True
                    # replicated param
            if not use_shard:
                param.set_dist_spec(ReplicaSpec())
            
            param.is_visited = True

    total_numel = 0
    for name, p in model.named_parameters():
        total_numel += p.numel()
    print(f"numel of the model {total_numel/1e9}")

    # model inference
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logits = outputs.logits
    
    torch.cuda.synchronize()
    print(logits)

if __name__ == '__main__':
    args = get_args()
    if args.backend == "colossalai":
        run_CAI(args)
    elif args.backend == "torch":
        run_torch(args)