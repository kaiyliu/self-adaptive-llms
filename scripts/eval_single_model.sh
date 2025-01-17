#!/bin/bash

# Task Selection
TASK="mbpp2" # Available options: mbpp2, math, ai2_arc

# First Stage Inference: Classification Expert
# Set to 'None' if not using cls expert
CLS_EXPERT_PATH="None"

# Second Stage: Expert Models
# Replace these paths with your actual model paths
ORI_MODEL_PATH="layoric/llama-2-13b-code-alpaca"

# Start evaluation!
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python svd_reinforce_hydra.py \
    base_model@_global_=llama3i8b \
    task@_global_=$TASK \
    mode@_global_=eval \
    prompt_based_eval=false \
    experts_path_dict.ori_model=$ORI_MODEL_PATH