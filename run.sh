#!/usr/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=6,7

# training 
# /home/yht/.conda/envs/sood-mcl run -n sood-mcl
cd /data/yht/code/HeltonRotation


# fcos
/home/yht/.conda/envs/sood-mcl/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port=29558 \
    runner.py \
    --config ./configs/fcos.py 