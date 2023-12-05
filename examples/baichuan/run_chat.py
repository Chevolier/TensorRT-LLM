# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip
from typing import List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm
import time

# from flask import Flask, request, jsonify

# app = Flask(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--model_version',
                        type=str,
                        default='v1_13b',
                        choices=['v1_7b', 'v1_13b', 'v2_7b', 'v2_13b'])
    parser.add_argument('--engine_dir', type=str, default='baichuan_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="baichuan-inc/Baichuan-13B-Chat",
                        help="Directory containing the tokenizer.model.")
    
    parser.add_argument('--hf_config_dir',
                type=str,
                default="baichuan-inc/Baichuan-13B-Chat",
                help="Directory containing the huggingface config file")
    
    parser.add_argument('--generation_config_dir',
                    type=str,
                    default="baichuan-inc/Baichuan-13B-Chat",
                    help="Directory containing the generation config file")

    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    return parser.parse_args()

def build_chat_input(config, generation_config, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or generation_config['max_new_tokens']
    max_input_tokens = config['model_max_length'] - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(generation_config['user_token_id'])
            else:
                round_tokens.append(generation_config['assistant_token_id'])
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(generation_config['assistant_token_id'])
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left

    input_ids = torch.tensor([input_tokens], dtype=torch.int32, device='cuda')
    input_lengths = torch.tensor([input_ids.size(1)],
                                    dtype=torch.int32,
                                     device='cuda')

    return input_ids, input_lengths

def generate(
    decoder,
    tokenizer,
    config,
    hf_config,
    generation_config,
    messages: List[dict],
    max_output_len: int = 1024,
    model_version: str = 'v2_13b',
    num_beams: int = 1,
):
    # tensorrt_llm.logger.set_level(log_level)

    runtime_rank = tensorrt_llm.mpi_rank()

    repetition_penalty = 1.3
    temperature = 0.3
    top_k = 5
    top_p = 0.85
    if model_version == 'v1_7b':
        temperature = 1
        top_k = 1
        top_p = 0
    elif model_version == 'v2_7b' or model_version == 'v2_13b':
        repetition_penalty = 1.05
    sampling_config = SamplingConfig(end_id=generation_config['eos_token_id'],
                                     pad_id=generation_config['pad_token_id'],
                                     num_beams=num_beams,
                                     repetition_penalty=repetition_penalty,
                                     temperature=temperature,
                                     top_k=top_k,
                                     top_p=top_p)

    input_ids, input_lengths = build_chat_input(hf_config,
                                                generation_config,
                                                tokenizer,
                                                messages,
                                                max_new_tokens=max_output_len)

    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0),
                  max_input_length,
                  max_output_len,
                  beam_width=num_beams)

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()
    
    output_text = ''
    if runtime_rank == 0:
        for b in range(input_lengths.size(0)):
            if num_beams <= 1:
                output_begin = max_input_length
                outputs = output_ids[b][0][output_begin:].tolist()
                output_text = tokenizer.decode(outputs, skip_special_tokens=True)
                # print(f'Output: \"{output_text}\"')
            else:
                for beam in range(num_beams):
                    output_begin = input_lengths[b]
                    output_end = input_lengths[b] + max_output_len
                    outputs = output_ids[b][beam][
                        output_begin:output_end].tolist()
                    output_text = tokenizer.decode(outputs, skip_special_tokens=True,)
                    # print(f'Output: \"{output_text}\"')

        output_ids = output_ids.reshape((-1, output_ids.size(2)))

    return output_text

# load the trt_engine
def load_engine(
    engine_dir: str = 'baichuan_outputs',
    tokenizer_dir: str = None,
    hf_config_dir: str = 'baichuan_outputs',
    generation_config_dir: str = 'baichuan_outputs',
    log_level: str = 'error',
):
    tensorrt_llm.logger.set_level(log_level)

    config_path = os.path.join(engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    hf_config_path = os.path.join(hf_config_dir, 'config.json')
    with open(hf_config_path, 'r') as f:
        hf_config = json.load(f)
    
    generation_config_path = os.path.join(generation_config_dir, 'generation_config.json')
    with open(generation_config_path, 'r') as f:
        generation_config = json.load(f)
        # generation_config['max_new_tokens'] = 1024
        
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                              use_fast=False,
                                              trust_remote_code=True)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               dtype=dtype)

    engine_name = get_engine_name('baichuan', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(engine_dir, engine_name)
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    return decoder, tokenizer, config, hf_config, generation_config

# load engine
args = parse_arguments()
decoder, tokenizer, config, hf_config, generation_config = load_engine(args.engine_dir,
                                                                args.tokenizer_dir,
                                                                args.hf_config_dir,
                                                                args.generation_config_dir,
                                                                args.log_level)

# @app.route('/invocations', methods=['POST'])
# def invocations():
#     prompt = request.json['prompt']
    
#     try:
#         messages = [{"role": "user", "content": prompt}]
#         response = generate(
#                         decoder,
#                         tokenizer,
#                         config,
#                         hf_config,
#                         generation_config,
#                         messages,
#                         max_output_len=1024
#                     )
        
#         return jsonify({'response': response, 'code': 200})

#     except Exception as e:
#         return jsonify({'error': str(e), 'code': 500})


# def get_cmd(world_size, tritonserver, model_repo):
#     cmd = 'mpirun --allow-run-as-root '
#     for i in range(world_size):
#         cmd += ' -n 1 {} --model-repository={} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix{}_ : '.format(
#             tritonserver, model_repo, i)
#     cmd += '&'
#     return cmd

# if __name__ == '__main__':
#     # args = parse_arguments()
#     # cmd = get_cmd(int(args.world_size), args.tritonserver, args.model_repo)
#     # subprocess.call(cmd, shell=True)
#     # app.run(debug=True)
#     pass
#     args = parse_arguments()
#     cmd = get_cmd(int(args.world_size), args.tritonserver, args.model_repo)
#     subprocess.call(cmd, shell=True)

if __name__ == '__main__':

    prompt = '请以《哇塞，杭州太好玩了》为题，写一篇小红书风格的旅游攻略文案，不少于500字。'
    messages = [{"role": "user", "content": prompt}]
    
    path_test = "test_gen_v6_trt.csv"
    df_test = pd.read_csv(path_test)
    
    for i, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        messages = [{"role": "user", "content": row['prompt']}]
        start_time = time.time()
        
        response = generate(
                        decoder,
                        tokenizer,
                        config,
                        hf_config,
                        generation_config,
                        messages,
                        max_output_len=1024
                    )

        time_cost = time.time() - start_time
        df_test.loc[i, 'response_trt_int8'] = response
        df_test.loc[i, 'time_cost_trt_int8'] = time_cost
            
    df_test.to_csv("./test_gen_v6_trt_int8.csv", index=False, encoding='utf_8_sig')
    
    # # Run the Flask app
    # app.run(debug=True)


