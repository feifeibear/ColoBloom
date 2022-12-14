## Inference solutions for BLOOM 176B

We support HuggingFace accelerate and DeepSpeed Inference for generation.

Install required packages:

```shell
pip install flask flask_api gunicorn pydantic accelerate huggingface_hub>=0.9.0 deepspeed>=0.7.3
```
To install [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII):
```shell
git clone https://github.com/microsoft/DeepSpeed-MII
cd DeepSpeed-MII
pip install .
```

All the provided scripts are tested on 8 A100 80GB GPUs for BLOOM 176B (fp16/bf16) and 4 A100 80GB GPUs for BLOOM 176B (int8). These scripts might not work for other models or a different number of GPUs.

DS inference is deployed using the DeepSpeed MII library which requires the resharded checkpoints for 8 x Tensor Parallel.

Note: sometimes GPU memory is not freed when DS inference deployment is shutdown. You can free this memory by running:
```python
import mii
mii.terminate("ds_inference_grpc_server")
```
or alternatively, just doing a `killall python` in terminal.

For using BLOOM quantized, use dtype = int8. Also, change the model_name to microsoft/bloom-deepspeed-inference-int8 for DeepSpeed-Inference. For HF accelerate, no change is needed for model_name.

HF accelerate uses [LLM.int8()](https://arxiv.org/abs/2208.07339) and DS-inference uses [ZeroQuant](https://arxiv.org/abs/2206.01861) for post-training quantization.

#### BLOOM inference via command-line

This asks for generate_kwargs everytime.
Example: generate_kwargs =
```json
{"min_length": 100, "max_new_tokens": 100, "do_sample": false}
```

1. using HF accelerate
```shell
python cli.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --generate_kwargs '{"min_length": 100, "max_new_tokens": 100, "do_sample": false}'
```

2. using DS inference
```shell
python cli.py --model_name microsoft/bloom-deepspeed-inference-fp16 --dtype fp16 --deployment_framework ds_inference --generate_kwargs '{"min_length": 100, "max_new_tokens": 100, "do_sample": false}'
```

#### BLOOM server deployment

[server.sh](server.sh) can be used to launch a generation server. Please note that the serving method is synchronous and users have to wait in queue until the preceding requests have been processed.

#### Benchmark system for BLOOM inference

1. using HF accelerate
```shell
python benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --benchmark_cycles 5
```

2. using DS inference
```shell
deepspeed --num_gpus 8 benchmark.py --model_name bigscience/bloom --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5
```
alternatively, to load model faster:
```shell
deepspeed --num_gpus 8 benchmark.py --model_name microsoft/bloom-deepspeed-inference-fp16 --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5
```

3. using DS ZeRO
```shell
deepspeed --num_gpus 8 benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework ds_zero --benchmark_cycles 5
```

## Support


If you run into things not working or have other questions please open an Issue in the corresponding backend:

- [Accelerate](https://github.com/huggingface/accelerate/issues)
- [Deepspeed-Inference](https://github.com/microsoft/DeepSpeed/issues)
- [Deepspeed-ZeRO](https://github.com/microsoft/DeepSpeed/issues)

If there a specific issue with one of the scripts and not the backend only then please open an Issue here and tag [@mayank31398](https://github.com/mayank31398).
