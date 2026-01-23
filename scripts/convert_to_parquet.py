#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Convert Hugging Face Arrow datasets to Parquet format for training.
This script converts ThinkMorph datasets from Arrow format to Parquet format
and generates the required parquet_info.json files.
"""

import os
import json
import argparse
import io
from pathlib import Path
from datasets import load_from_disk
import pyarrow.parquet as pq
from PIL import Image


def image_to_bytes(pil_image):
    """Convert PIL Image to bytes."""
    if pil_image is None:
        return None
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def reformat_thinkmorph_sample(sample):
    """
    Reformat ThinkMorph dataset sample to match the expected format.

    Input format:
        - question, answer, problem_image_0, reasoning_image_0,
          resoning_thought_0, resoning_thought_1

    Output format:
        - instruction_list: [question]
        - image_list: [problem_image_0, reasoning_image_0]
        - output_text_list: [
            f"<think>{resoning_thought_0}</think><image_start>",
            f"<image_end><think>{resoning_thought_1}</think><answer>{answer}</answer>"
          ]
    """
    # Convert PIL images to bytes
    problem_image_bytes = image_to_bytes(sample['problem_image_0'])
    reasoning_image_bytes = image_to_bytes(sample['reasoning_image_0'])

    return {
        'instruction_list': [sample['question']],
        'image_list': [problem_image_bytes, reasoning_image_bytes],
        'output_text_list': [
            f"<think>{sample['resoning_thought_0']}</think><image_start>",
            f"<image_end><think>{sample['resoning_thought_1']}</think><answer>{sample['answer']}</answer>"
        ]
    }


def convert_dataset_to_parquet(arrow_dir, parquet_dir, num_shards=3, rows_per_shard=2000):
    """
    Convert a Hugging Face Arrow dataset to Parquet format.
    
    Args:
        arrow_dir: Path to the Arrow dataset directory
        parquet_dir: Path to save Parquet files
        num_shards: Number of parquet files to create
        rows_per_shard: Approximate number of rows per shard
    """
    print(f"\n{'='*80}")
    print(f"Converting: {arrow_dir}")
    print(f"Output: {parquet_dir}")
    print(f"{'='*80}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(arrow_dir)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")

    # Reformat dataset to match expected format
    print("Reformatting dataset...")
    reformatted_data = []
    for idx, sample in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{total_samples} samples...")
        try:
            reformatted_sample = reformat_thinkmorph_sample(sample)
            reformatted_data.append(reformatted_sample)
        except Exception as e:
            print(f"  Warning: Failed to reformat sample {idx}: {e}")
            continue

    print(f"Successfully reformatted {len(reformatted_data)}/{total_samples} samples")

    # Create output directory
    os.makedirs(parquet_dir, exist_ok=True)

    # Calculate shard sizes
    total_reformatted = len(reformatted_data)
    shard_size = (total_reformatted + num_shards - 1) // num_shards
    print(f"Creating {num_shards} shards with ~{shard_size} samples each")

    parquet_info = {}

    # Convert to parquet shards
    import pandas as pd
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, total_reformatted)

        if start_idx >= total_reformatted:
            break

        shard_data = reformatted_data[start_idx:end_idx]
        parquet_filename = f"data_{shard_idx:05d}.parquet"
        parquet_file = os.path.join(parquet_dir, parquet_filename)

        print(f"  Shard {shard_idx}: samples {start_idx}-{end_idx} -> {parquet_file}")

        # Convert to pandas and save as parquet
        df = pd.DataFrame(shard_data)
        df.to_parquet(parquet_file, engine='pyarrow', index=False)

        # Get parquet metadata
        parquet_file_obj = pq.ParquetFile(parquet_file)
        num_row_groups = parquet_file_obj.num_row_groups

        # Store parquet info with absolute path
        parquet_info[os.path.abspath(parquet_file)] = {
            'num_row_groups': num_row_groups,
            'num_rows': end_idx - start_idx
        }

        print(f"    ✓ Created with {num_row_groups} row groups, {end_idx - start_idx} rows")
    
    # Save parquet_info.json
    info_file = os.path.join(parquet_dir, "parquet_info.json")
    with open(info_file, 'w') as f:
        json.dump(parquet_info, f, indent=2)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Parquet files: {parquet_dir}")
    print(f"  Info file: {info_file}")
    
    return parquet_dir, info_file


def main():
    parser = argparse.ArgumentParser(description="Convert Arrow datasets to Parquet format")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/training",
        help="Directory containing Arrow datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training_parquet",
        help="Directory to save Parquet datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Jigsaw_Assembly", "Spatial_Navigation", "Visual_Search", "Chart_Refocus"],
        help="List of dataset names to convert"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=3,
        help="Number of parquet shards per dataset"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ThinkMorph Dataset Converter: Arrow → Parquet")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Shards per dataset: {args.num_shards}")
    
    # Convert each dataset
    results = {}
    for dataset_name in args.datasets:
        arrow_dir = os.path.join(args.input_dir, dataset_name)
        parquet_dir = os.path.join(args.output_dir, dataset_name)
        
        if not os.path.exists(arrow_dir):
            print(f"\n⚠ Warning: {arrow_dir} does not exist, skipping...")
            continue
        
        try:
            parquet_path, info_path = convert_dataset_to_parquet(
                arrow_dir, parquet_dir, num_shards=args.num_shards
            )
            results[dataset_name] = {
                'parquet_dir': parquet_path,
                'info_file': info_path
            }
        except Exception as e:
            print(f"\n✗ Error converting {dataset_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("Conversion Summary")
    print("="*80)
    for dataset_name, paths in results.items():
        print(f"✓ {dataset_name}")
        print(f"    Parquet: {paths['parquet_dir']}")
        print(f"    Info: {paths['info_file']}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("Update data/dataset_info.py with the new parquet paths:")
    print(f"  'data_dir': '{args.output_dir}/<dataset_name>'")
    print(f"  'parquet_info_path': '{args.output_dir}/<dataset_name>/parquet_info.json'")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

