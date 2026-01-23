# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# ThinkMorph Full Training Script
# Checkpoints saved to: ckpt/thinkmorph/<run_name>/

# Set default paths (can be overridden by environment variables)
resume_from=${resume_from:-"models/BAGEL-7B-MoT"}
run_name=${run_name:-"run_8gpu"}
output_path=${output_path:-"./ckpt/thinkmorph/${run_name}/output"}
ckpt_path=${ckpt_path:-"./ckpt/thinkmorph/${run_name}"}

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/interleaved_reasoning.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --finetune-from-ema True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --lr 1e-5 \
  --num_worker 4 \
  --max_latent_size 64  \
  --max_num_tokens 32768 \
  --mse_weight 1 \
  --ce_weight 1 \
  --total_steps 8000 \