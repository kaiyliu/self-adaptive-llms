# !/bin/bash

for TASK in moe; do
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
    echo "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python moe_svd_reinforce_hydra.py \
        base_model@_global_=llama3i8b \
        task@_global_=$TASK \
        policy@_global_=wcomb \
        optimization@_global_=reinforce \
        mode@_global_=training \
        num_iters=$NUM_ITERS \
        kl_ref_coeff=0.0"
done


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python moe_svd_reinforce_hydra.py \
    base_model@_global_=llama3i8b \
    task@_global_=moe \
    policy@_global_=wcomb \
    optimization@_global_=reinforce \
    mode@_global_=training \
    num_iters=200 \
    kl_ref_coeff=0.0 \
    batch_size=10
    