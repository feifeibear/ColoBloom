import torch
from transformers import BloomTokenizerFast, BloomForCausalLM, BloomConfig
import torch.distributed as dist
import os
from utils import replace_8bit_linear_tp, get_8bit_tp_model, get_8bit_tp_model_list, replace_8bit_linear, getModelSize, init_empty_weights, Linear8bitTP
import time
import torch.profiler

datapath = "/data2/users/lccsr/bloom3b/data"     

def add_param(model, param_tensor, name):
        param = torch.nn.Parameter(param_tensor, requires_grad=False)
        name_list = name.split('.')
        module = model._modules[name_list[0]]
        for i in range(1, len(name_list) - 1):
                module = module._modules[name_list[i]]
        module._parameters[name_list[-1]] = param
        del param_tensor
        return model

def run_int8():
        world_size = torch.cuda.device_count()
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        rank = local_rank
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        cpu_group = dist.new_group(backend='gloo')

        # meta init
        if rank == 0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
                profile_memory=True, 
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
                record_shapes=True,
                with_stack=True
                ) as prof:        
                        model = BloomForCausalLM.from_pretrained(datapath, low_cpu_mem_usage=True).half()
                        # get meta_model_list
                        model_list = []
                        with init_empty_weights():
                                configuration = BloomConfig.from_json_file(datapath + '/config.json')
                                for i in range(world_size):
                                        model_list.append(BloomForCausalLM(configuration).half())
                        # get quant & sharded model_list
                        model_list = get_8bit_tp_model_list(model, model_list, world_size)
                        # send out
                        for name, param in model_list[0].named_parameters():
                                param_list = [param.data]
                                for i in range(1, world_size):
                                        param_list.append(model_list[i].state_dict()[name])
                                param_tensor = torch.zeros_like(param_list[0],dtype=param_list[0].dtype)
                                dist.scatter(param_tensor, scatter_list=param_list, src=0, group=cpu_group)
                                del param_list, param_tensor
                        model = model_list[0]
                        del model_list
                        prof.step()
                        

        else:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
                profile_memory=True, 
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
                record_shapes=True,
                with_stack=True
                ) as prof:
                        with init_empty_weights():
                                configuration = BloomConfig.from_json_file(datapath + '/config.json')
                                model = BloomForCausalLM(configuration).half()
                        model = get_8bit_tp_model(model, rank, world_size)
                        for name, param in model.named_parameters():
                                param_tensor = torch.zeros(param.data.size(), dtype=param.dtype)
                                dist.scatter(param_tensor, src=0, group=cpu_group)
                                model = add_param(model, param_tensor, name)
                        model._modules['lm_head']._parameters['weight']= model._modules['transformer']._modules['word_embeddings'].weight  
                        prof.step()
                
        
        model = model.to(rank)
        
        # generate inference
        tokenizer = BloomTokenizerFast.from_pretrained(datapath)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(rank)
        generate_kwargs = dict(max_new_tokens=100, do_sample=False)
        outputs = model.generate(**inputs, **generate_kwargs)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(rank, outputs)
                

    
def run_fp16():
        model = BloomForCausalLM.from_pretrained(datapath, low_cpu_mem_usage=True).half().to(0)
        tokenizer = BloomTokenizerFast.from_pretrained(datapath)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(0)
        generate_kwargs = dict(max_new_tokens=100, do_sample=False)
        outputs = model.generate(**inputs, **generate_kwargs)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(outputs)

        
if __name__ == '__main__':
        # run_fp16()
        run_int8()
