#!/bin/bash

resume_from=${resume_from:-"models/BAGEL-7B-MoT"}
output_path=${output_path:-"./output_pat_2gpu"}
ckpt_path=${ckpt_path:-"./ckpt_pat_2gpu"}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=2 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/path_tracing_2gpu.yaml \
  --model_path $resume_from \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune-from-ema True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --lr 1e-5 \
  --num_workers 4 \
  --max_latent_size 64 \
  --max_num_tokens 16384 \
  --max_num_tokens_per_sample 12288 \
  --expected_num_tokens 12288 \
  --mse_weight 1 \
  --ce_weight 1 \
  --num_shard 2 \
  --total_steps 50 \
  --log_every 5 \
  --save_every 25

