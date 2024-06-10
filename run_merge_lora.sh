#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1
llamafactory-cli export examples/merge_lora/llama3_lora_sft_merge_catchwords.yaml