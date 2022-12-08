import torch
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig
import torch.distributed as dist
import os
from utils import  init_dtype_weights, get_8bit_tp_model, get_8bit_tp_model_list, getModelSize, init_empty_weights, replace_8bit_linear_tp
import time
import copy
import torch.profiler
import datetime
import colossalai
import contextlib


def add_param(model, param_tensor, name):
    param = torch.nn.Parameter(param_tensor, requires_grad=False)
    name_list = name.split('.')
    module = model._modules[name_list[0]]
    for i in range(1, len(name_list) - 1):
        module = module._modules[name_list[i]]
    module._parameters[name_list[-1]] = param
    del param_tensor
    return model


def run_int8_bloom_inference(from_pretrain=False, data_path=None, use_profiler=False):
    colossalai.launch_from_torch(config={})
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cpu_group = dist.new_group(
        backend='gloo', timeout=datetime.timedelta(seconds=18000))
    if from_pretrain:
        configuration = BloomConfig.from_json_file(data_path + '/config.json')
    else:
        configuration = BloomConfig(
            hidden_size=14336,
            n_layer=70,
            n_head=112,)

    if use_profiler:
        ctx = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                     profile_memory=True,
                                     on_trace_ready=torch.profiler.tensorboard_trace_handler(
                                         './log'),
                                     )
    else:
        ctx = contextlib.nullcontext()

    with ctx as prof:
        # meta init
        time0 = time.time()
        if rank == 0:
            # get meta_model_list
            model_list = []
            with init_empty_weights():
                model_tmp = BloomForCausalLM(configuration).half()

            model_list.append(model_tmp)
            for i in range(world_size):
                model_list.append(copy.deepcopy(model_tmp))

            # get pre_trained model
            if from_pretrain:
                model = BloomForCausalLM.from_pretrained(
                    data_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
            else:
                with init_dtype_weights(dtype=torch.float16, use_skip_init=True):
                    model = BloomForCausalLM(configuration)
            getModelSize(model)
            print("Model load complete", time.time() - time0)

            # get quant & sharded model_list
            model_list = get_8bit_tp_model_list(model, model_list, world_size)
            print("Model init complete", time.time() - time0)

            dist.barrier(cpu_group)
            # send out
            for name, param in model_list[0].named_parameters():
                param_list = [param.data]
                for i in range(1, world_size):
                    param_list.append(model_list[i].state_dict()[name])
                param_tensor = torch.zeros_like(
                    param_list[0], dtype=param_list[0].dtype)
                dist.scatter(param_tensor, scatter_list=param_list,
                             src=0, group=cpu_group)
                del param_list, param_tensor
            model = model_list[0]
            del model_list

        else:
            with init_empty_weights():
                model = BloomForCausalLM(configuration).half()

            print("rank: ", rank, ", model load complete")
            model = get_8bit_tp_model(model, rank, world_size)
            print("rank: ", rank, ", model init complete")

            dist.barrier(cpu_group)
            for name, param in model.named_parameters():
                param_tensor = torch.zeros(
                    param.data.size(), dtype=param.dtype)
                dist.scatter(param_tensor, src=0, group=cpu_group)
                model = add_param(model, param_tensor, name)
            model._modules['lm_head']._parameters['weight'] = model._modules['transformer']._modules['word_embeddings'].weight
            # pass

        time2 = time.time()
        print("Model trans complete", time2 - time0, "s")
        model = model.to(rank)
        getModelSize(model)

        tokenizer = BloomTokenizerFast.from_pretrained(data_path)
        inputs = tokenizer("Hello, my dog is cute",
                           return_tensors="pt").to(rank)
        # warm up
        for _ in range(10):
            output = model(**inputs, labels=inputs["input_ids"])
        # inference
        time_list = []
        for _ in range(20):
            timet = time.time()
            output = model(**inputs, labels=inputs["input_ids"])
            time_list.append(time.time() - timet)

        print("avg inference latency:", sum(time_list)/20)
        print("Max GPU mem allocated:", torch.cuda.max_memory_allocated(rank))
        if use_profiler:
            prof.step()


def run_fp16(from_pretrain=False, data_path=None):
    if from_pretrain:
        model = BloomForCausalLM.from_pretrained(
            data_path, low_cpu_mem_usage=True).half().to(0)
    else:
        cfg = BloomConfig(
            hidden_size=14336,
            n_layer=70,
            n_head=112,)
        with init_dtype_weights(dtype=torch.float16):
            with skip_init():
                model = BloomForCausalLM(cfg)
        model = model.to(0)
    tokenizer = BloomTokenizerFast.from_pretrained(data_path)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(0)
    generate_kwargs = dict(max_new_tokens=100, do_sample=False)
    outputs = model.generate(**inputs, **generate_kwargs)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(outputs)


def run(int8=True, from_pretrain=False, data_path=None, profile=False):
    if int8:
        run_int8_bloom_inference(from_pretrain, data_path, profile)
    else:
        run_fp16(from_pretrain, data_path)


if __name__ == '__main__':
    int8 = True
    from_pretrain = True
    data_path = "/data2/users/lczht/bloom-560m"
    profile = False
    run(int8, from_pretrain, data_path, profile)
