#!/usr/bin/env bash

# CUR_DIR=$(cd $(dirname $0); pwd)
# cd $CUR_DIR

# 取 worker0 第一个 port
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

if [ -z "$PRETRAIN_PATH" ]; then
    PRETRAIN_PATH="/mnt/bn/chengru-nas/models/llava-v1.6-mistral-7b"
fi

if [ -z "$DATA_PATH" ]; then
    DATA_PATH="/mnt/bn/chengru-nas/train_data/llava/20240412/llava_projector_tune/train.json"
fi

if [ -z "$VISION_TOWER" ]; then
    VISION_TOWER="/mnt/bn/chengru-nas/models/clip-vit-large-patch14-336"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/mnt/bn/chengru-nas/models/ckpts/llava_mistral/20240412/llava_projector_tune"
fi

if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=1
fi

if [ -z "$NUM_EPOCHS" ]; then
    NUM_EPOCHS=5
fi
#原始的脚本启动方式
#python3 train.py $@

echo "pretrain path: $PRETRAIN_PATH"
echo "data path: $DATA_PATH"
echo "vision tower: $VISION_TOWER"
echo "output dir: $OUTPUT_DIR"
echo "batch size: $BATCH_SIZE" 
echo "num epochs: $NUM_EPOCHS"

bash scripts/v1_5/bootstrap.sh train \
llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $PRETRAIN_PATH \
    --version v1 \
    --data_path $DATA_PATH \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
# 使用 torch.distributed.launch 启动
# torchrun \
# --nnodes $ARNOLD_WORKER_NUM \
# --node_rank $ARNOLD_ID \
# --nproc_per_node $ARNOLD_WORKER_GPU \
# --master_addr $METIS_WORKER_0_HOST \
# --master_port $PORT0 \
# llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path $PRETRAIN_PATH \
#     --version v1 \
#     --data_path $DATA_PATH \
#     --vision_tower $VISION_TOWER \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir $OUTPUT_DIR \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size $BATCH_SIZE \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True
    #--pretrain_mm_mlp_adapter /mnt/bn/chengru-nas/models/llava1_5/mm_projector.bin \