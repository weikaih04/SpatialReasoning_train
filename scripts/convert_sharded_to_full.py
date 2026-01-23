#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""
Convert sharded FSDP checkpoint to full model for evaluation.

This script converts a checkpoint saved with SHARDED_STATE_DICT format
(model/, ema/ directories) to the full model format (model.safetensors)
that can be loaded by the evaluation code.

Usage:
    python scripts/convert_sharded_to_full.py \
        --checkpoint_path ckpt/multi_view_counting/run_8gpu/0001000 \
        --output_path ckpt/multi_view_counting/run_8gpu/0001000_full \
        --model_path models/BAGEL-7B-MoT \
        --convert_ema  # Optional: convert EMA instead of model

Reference: https://github.com/ByteDance-Seed/Bagel/issues/139
"""

import argparse
import gc
import os
import shutil
import sys

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, load_state_dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.autoencoder import load_ae


def load_model_architecture(model_path, layer_module="Qwen2MoTDecoderLayer", visual_gen=True):
    """Load model architecture from HuggingFace format without weights."""

    # Load LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.layer_module = layer_module
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    language_model = Qwen2ForCausalLM(llm_config)

    # Load ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 - 2  # vit_select_layer=-2
    vit_config.rope = False
    vit_model = SiglipVisionModel(vit_config)

    # Load VAE config
    _, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Create Bagel config
    config = BagelConfig(
        visual_gen=visual_gen,  # Can be False for no_thought models
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,  # Use 64 for compatibility
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
    )

    # Create model
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    return model


def convert_sharded_to_full(
    checkpoint_path: str,
    output_path: str,
    model_path: str,
    convert_ema: bool = False,
    layer_module: str = "Qwen2MoTDecoderLayer",
    visual_gen: bool = True,
):
    """Convert sharded checkpoint to full model."""

    # Validate input
    model_dir = os.path.join(checkpoint_path, "model")
    ema_dir = os.path.join(checkpoint_path, "ema")

    if convert_ema:
        source_dir = ema_dir
        output_name = "ema.safetensors"
    else:
        source_dir = model_dir
        output_name = "model.safetensors"

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        print("This checkpoint may already be in full format.")
        return False

    print(f"Converting sharded checkpoint from: {source_dir}")
    print(f"Output path: {output_path}")
    print(f"Model architecture from: {model_path}")
    print(f"Visual generation: {visual_gen}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load model architecture (without weights)
    print("Loading model architecture...")
    model = load_model_architecture(model_path, layer_module, visual_gen=visual_gen)

    # Load sharded checkpoint into model
    print("Loading sharded checkpoint...")
    model_state_dict = model.state_dict()
    reader = FileSystemReader(source_dir)
    load_state_dict(model_state_dict, reader, no_dist=True)
    model.load_state_dict(model_state_dict, strict=False)
    del model_state_dict
    gc.collect()

    # Extract state dict
    print("Extracting full state dict...")
    full_state_dict = model.state_dict()

    # Save as safetensors
    output_file = os.path.join(output_path, output_name)
    print(f"Saving to: {output_file}")
    save_file(full_state_dict, output_file)

    # Report size
    file_size = os.path.getsize(output_file) / 1e9
    print(f"Saved {output_name}: {file_size:.2f} GB")

    del full_state_dict
    gc.collect()

    # Copy config files from model_path
    print("Copying config files...")
    config_files = [
        "config.json",
        "generation_config.json",
        "llm_config.json",
        "vit_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "ae.safetensors",
    ]

    for config_file in config_files:
        src = os.path.join(model_path, config_file)
        dst = os.path.join(output_path, config_file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"  Copied: {config_file}")

    print(f"\nConversion complete!")
    print(f"Full checkpoint saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert sharded FSDP checkpoint to full model for evaluation"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the sharded checkpoint directory (e.g., ckpt/xxx/0001000)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the full checkpoint (e.g., ckpt/xxx/0001000_full)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/BAGEL-7B-MoT",
        help="Path to base model for architecture (default: models/BAGEL-7B-MoT)"
    )
    parser.add_argument(
        "--convert_ema",
        action="store_true",
        help="Convert EMA model instead of training model"
    )
    parser.add_argument(
        "--layer_module",
        type=str,
        default="Qwen2MoTDecoderLayer",
        help="Layer module class name (default: Qwen2MoTDecoderLayer)"
    )
    parser.add_argument(
        "--no_visual_gen",
        action="store_true",
        help="Use visual_gen=False for models trained without image generation (e.g., no_thought baseline)"
    )

    args = parser.parse_args()

    success = convert_sharded_to_full(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        model_path=args.model_path,
        convert_ema=args.convert_ema,
        layer_module=args.layer_module,
        visual_gen=not args.no_visual_gen,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
