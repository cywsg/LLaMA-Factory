#!/bin/sh

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config_file examples/accelerate/single_config_catchwords.yaml \
    src/train.py examples/lora_multi_gpu/llama3_lora_sft_catchwords.yaml > log_lora_all_2 2>&1

# accelerate launch \
#     --config_file examples/accelerate/single_config_catchwords.yaml \
#     src/train.py examples/lora_multi_gpu/llama3_lora_sft_catchwords.yaml 


# # DeepSpeed
# NPROC_PER_NODE=4

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
#     --nproc_per_node $NPROC_PER_NODE \
#     --nnodes 1 \
#     --standalone \
    # src/train.py examples/lora_multi_gpu/llama3_lora_sft_catchwords_ds.yaml