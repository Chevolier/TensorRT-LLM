# Baichuan

This document shows how to build and run a Baichuan models (including `v1_7b`/`v1_13b`/`v2_7b`/`v2_13b`) in TensorRT-LLM on both single GPU and single node multi-GPU.

## Overview

The TensorRT-LLM Baichuan implementation can be found in [tensorrt_llm/models/baichuan/model.py](../../tensorrt_llm/models/baichuan/model.py). The TensorRT-LLM Baichuan example code is located in [`examples/baichuan`](./). There are three main files in that folder::

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Baichuan model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

These scripts accept an argument named model_version, whose value should be `v1_7b`/`v1_13b`/`v2_7b`/`v2_13b` and the default value is `v1_13b`.

## Support Matrix
  * FP16
  * BF16
  * INT4 & INT8 Weight-Only

## Usage

The TensorRT-LLM Baichuan example code locates at [examples/baichuan](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to specify the HF Baichuan checkpoint path. For `v1_13b`, you should use whether [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) or [baichuan-inc/Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base). For `v2_13b`, you should use whether [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) or [baichuan-inc/Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base). More Baichuan models could be found on [baichuan-inc](https://huggingface.co/baichuan-inc).

TensorRT-LLM Baichuan builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples that take `v1_13b` as example:

```bash
# Build a single-GPU float16 engine from HF weights.
# Enable the special TensorRT-LLM GPT Attention plugin (--use_gpt_attention_plugin) to increase runtime performance.
# 7B models should always add --use_gpt_attention_plugin since RoPE is only supported with GPTAttention plugin now.
# Try use_gemm_plugin to prevent accuracy issue.

##################
# A workable way for baichuan2

# Build the Baichuan V2 13B model using 4 GPUs and FP16.
python build.py --model_version v2_13b \
                --model_dir /code/tensorrt_llm/models/v3 \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./tmp/baichuan_v2_13b/trt_engines/fp16/4-gpu/ \
                --world_size 4

#                 --use_layernorm_plugin float16 \
#                 --hidden_act gelu \
#                 --remove_input_padding \
#                 --parallel_build \

# --parallel_build for world size 4 does not work
# Build the Baichuan V2 13B model using 4 GPUs and FP16, TensorRT engines with inflight_batching capability
python3 build.py --model_version v2_13b \
                 --model_dir=/code/tensorrt_llm/models/v6 \
                 --world_size=2 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gemm_plugin float16 \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --output_dir=./tmp/baichuan_v2_13b/trt_engines/v6/fp16-inflight/2-gpu/

# Build the Baichuan V2 13B model using 4 GPUs and FP16, TensorRT engines with inflight_batching capability
# remove paged_kv_cache and inflight_batching
python3 build.py --model_version v2_13b \
                 --model_dir=/code/tensorrt_llm/models/v6 \
                 --world_size=4 \
                 --dtype float16 \
                 --use_gemm_plugin float16 \
                 --use_gpt_attention_plugin float16 \
                 --output_dir=./tmp/baichuan_v2_13b/trt_engines/v6/fp16/4-gpu/


# Build TensorRT engines with INT8
python3 build.py --model_version v2_13b \
                 --model_dir=/code/tensorrt_llm/models/v6 \
                 --world_size=2 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gemm_plugin float16 \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_weight_only \
                 --output_dir=./tmp/baichuan_v2_13b/trt_engines/v6/int8-inflight/2-gpu/

# Build TensorRT engines with INT8, 1 GPU 
python3 build.py --model_version v2_13b \
                 --model_dir=/code/tensorrt_llm/models/v6 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gemm_plugin float16 \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_weight_only \
                 --output_dir=./tmp/baichuan_v2_13b/trt_engines/v6/int8-inflight/1-gpu/

# Build TensorRT engines with INT8
# remove paged_kv_cache and inflight_batching
python3 build.py --model_version v2_13b \
                 --model_dir=/code/tensorrt_llm/models/v6 \
                 --world_size=2 \
                 --dtype float16 \
                 --use_gemm_plugin float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_weight_only \
                 --parallel_build \
                 --output_dir=./tmp/baichuan_v2_13b/trt_engines/v6/int8/2-gpu/

# With 4-way tensor parallelism inference
mpirun -n 4 --allow-run-as-root \
    python run.py --model_version v2_13b \
                  --max_output_len=50 \
                  --tokenizer_dir=/code/tensorrt_llm/models/v3 \
                  --engine_dir=./tmp/baichuan_v2_13b/trt_engines/fp16/4-gpu/ 
                  
mpirun -n 4 --allow-run-as-root \
    python summarize.py --model_version v2_13b \
                        --test_trt_llm \
                        --hf_model_location /code/tensorrt_llm/models/v3 \
                        --data_type fp16 \
                        --engine_dir ./tmp/baichuan_v2_13b/trt_engines/fp16/4-gpu/ 
 
## test run_chat
mpirun -n 4 --allow-run-as-root \
    python run_chat.py --model_version v2_13b \
                        --hf_config_dir /code/tensorrt_llm/models/v6 \
                        --generation_config_dir /code/tensorrt_llm/models/v6 \
                        --engine_dir ./tmp/baichuan_v2_13b/trt_engines/v6/fp16-inflight/4-gpu/ \
                        --tokenizer_dir /code/tensorrt_llm/models/v6 \
                        --max_output_len 1024

mpirun -n 4 --allow-run-as-root \
    python run_chat.py --model_version v2_13b \
                        --hf_config_dir /code/tensorrt_llm/models/v6 \
                        --generation_config_dir /code/tensorrt_llm/models/v6 \
                        --engine_dir ./tmp/baichuan_v2_13b/trt_engines/v6/fp16/4-gpu/ \
                        --tokenizer_dir /code/tensorrt_llm/models/v6 \
                        --max_output_len 1024
                        
mpirun -n 2 --allow-run-as-root \
    python run_chat.py --model_version v2_13b \
                        --hf_config_dir /code/tensorrt_llm/models/v6 \
                        --generation_config_dir /code/tensorrt_llm/models/v6 \
                        --engine_dir ./tmp/baichuan_v2_13b/trt_engines/v6/fp16-inflight/2-gpu/ \
                        --tokenizer_dir /code/tensorrt_llm/models/v6 \
                        --max_output_len 1024                        
    
    
mpirun -n 2 --allow-run-as-root \
    python run_chat.py --model_version v2_13b \
                        --hf_config_dir /code/tensorrt_llm/models/v6 \
                        --generation_config_dir /code/tensorrt_llm/models/v6 \
                        --engine_dir ./tmp/baichuan_v2_13b/trt_engines/v6/int8/2-gpu/ \
                        --tokenizer_dir /code/tensorrt_llm/models/v6 \
                        --max_output_len 1024

mpirun -n 1 --allow-run-as-root \
    python run_chat.py --model_version v2_13b \
                        --hf_config_dir /code/tensorrt_llm/models/v6 \
                        --generation_config_dir /code/tensorrt_llm/models/v6 \
                        --engine_dir ./tmp/baichuan_v2_13b/trt_engines/v6/int8-inflight/1-gpu/ \
                        --tokenizer_dir /code/tensorrt_llm/models/v6 \
                        --max_output_len 1024
                        
#### INT8
# Build the Baichuan V2 13B model using a 2 GPUs and apply INT8 weight-only quantization.
CUDA_VISIBLE_DEVICES=0,1 python build.py --model_version v2_13b \
                --model_dir /code/tensorrt_llm/models/Baichuan2-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --use_weight_only \
                --output_dir ./tmp/baichuan_v2_13b/trt_engines/int8_weight_only/2-gpu/ \
                --world_size 2
                
# With 2-way tensor parallelism inference
mpirun -n 2 --allow-run-as-root \
    python run.py --model_version v2_13b \
                  --max_output_len=50 \
                  --tokenizer_dir=/code/tensorrt_llm/models/Baichuan2-13B-Chat \
                  --engine_dir=./tmp/baichuan_v2_13b/trt_engines/int8_weight_only/2-gpu/
                  
mpirun -n 2 --allow-run-as-root \
    python summarize.py --model_version v2_13b \
                        --test_trt_llm \
                        --hf_model_location /code/tensorrt_llm/models/Baichuan2-13B-Chat \
                        --data_type fp16 \
                        --engine_dir ./tmp/baichuan_v2_13b/trt_engines/int8_weight_only/2-gpu/

###################

# Build the Baichuan V1 13B model using a single GPU and FP16.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and BF16.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype bfloat16 \
                --use_gemm_plugin bfloat16 \
                --use_gpt_attention_plugin bfloat16 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/bf16/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and apply INT8 weight-only quantization.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --use_weight_only \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# Build the Baichuan V1 13B model using a single GPU and apply INT4 weight-only quantization.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/int4_weight_only/1-gpu/

# Build Baichuan V1 13B using 2-way tensor parallelism.
python build.py --model_version v1_13b \
                --model_dir baichuan-inc/Baichuan-13B-Chat \
                --dtype float16 \
                --use_gemm_plugin float16 \
                --use_gpt_attention_plugin float16 \
                --output_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/2-gpu/ \
                --world_size 2
```

                
### Run

To run a TensorRT-LLM Baichuan model using the engines generated by build.py

```bash
# With fp16 inference
python run.py --model_version v1_13b \
              --max_output_len=50 \
              --tokenizer_dir baichuan-inc/Baichuan-13B-Chat \
              --engine_dir=./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/

# With bf16 inference
python run.py --model_version v1_13b \
              --max_output_len=50 \
              --tokenizer_dir baichuan-inc/Baichuan-13B-Chat \
              --engine_dir=./tmp/baichuan_v1_13b/trt_engines/bf16/1-gpu/

# With INT8 weight-only quantization inference
python run.py --model_version v1_13b \
              --max_output_len=50 \
              --tokenizer_dir=baichuan-inc/Baichuan-13B-Chat \
              --engine_dir=./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# With INT4 weight-only quantization inference
python run.py --model_version v1_13b \
              --max_output_len=50 \
              --tokenizer_dir=baichuan-inc/Baichuan-13B-Chat \
              --engine_dir=./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# With 2-way tensor parallelism inference
mpirun -n 2 --allow-run-as-root \
    python run.py --model_version v1_13b \
                  --max_output_len=50 \
                  --tokenizer_dir=baichuan-inc/Baichuan-13B-Chat \
                  --engine_dir=./tmp/baichuan_v1_13b/trt_engines/fp16/2-gpu/



```



### Summarization using the Baichuan model

```bash
# Run summarization using the Baichuan V1 13B model in FP16.
python summarize.py --model_version v1_13b \
                    --test_trt_llm \
                    --hf_model_location baichuan-inc/Baichuan-13B-Chat \
                    --data_type fp16 \
                    --engine_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/1-gpu/

# Run summarization using the Baichuan V1 13B model quantized to INT8.
python summarize.py --model_version v1_13b \
                    --test_trt_llm \
                    --hf_model_location baichuan-inc/Baichuan-13B-Chat \
                    --data_type fp16 \
                    --engine_dir ./tmp/baichuan_v1_13b/trt_engines/int8_weight_only/1-gpu/

# Run summarization using the Baichuan V1 13B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --model_version v1_13b \
                        --test_trt_llm \
                        --hf_model_location baichuan-inc/Baichuan-13B-Chat \
                        --data_type fp16 \
                        --engine_dir ./tmp/baichuan_v1_13b/trt_engines/fp16/2-gpu/


```

### Known Issues

 * The implementation of the Baichuan-7B model with INT8 Weight-Only and Tensor
   Parallelism greater than 2 might have accuracy issues. It is under
   investigation.
