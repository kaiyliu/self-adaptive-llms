# !/bin/bash

for TASK in gsm8k; do
    # Task Selection
    # TASK="mbpp2" # Available options: mbpp2, gsm8k, ai2_arc, cls

    # Training Setting
    NUM_ITERS=200

    # if [ $TASK == "cls" ]; then
    #     # This script needs 2 gpus
    #     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python svd_reinforce_hydra.py \
    #         base_model@_global_=llama3i8b \
    #         task@_global_=$TASK \
    #         mode@_global_=training \
    #         num_iters=$NUM_ITERS \
    #         kl_ref_coeff=0.0
    # fi
    # This script needs 2 gpus
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python svd_reinforce_hydra.py \
        base_model@_global_=llama3i8b \
        task@_global_=$TASK \
        mode@_global_=training \
        num_iters=$NUM_ITERS \
        kl_ref_coeff=0.0
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python svd_reinforce_hydra.py \
        base_model@_global_=llama3i8b \
        task@_global_=$TASK \
        mode@_global_=training \
        num_iters=$NUM_ITERS \
        kl_ref_coeff=0.1
    # This script needs 2 gpus
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python svd_reinforce_hydra.py \
        base_model@_global_=llama3i8b \
        task@_global_=$TASK \
        mode@_global_=training \
        num_iters=$NUM_ITERS \
        kl_ref_coeff=0.2
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python svd_reinforce_hydra.py \
        base_model@_global_=llama3i8b \
        task@_global_=$TASK \
        mode@_global_=training \
        num_iters=$NUM_ITERS \
        kl_ref_coeff=0.3
done
