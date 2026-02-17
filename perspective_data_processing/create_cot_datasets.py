#!/usr/bin/env python3
"""
Create MM CoT and Text CoT Datasets for Perspective Taking.

Uses datasets.map() with num_proc for parallel processing (same as transform_perspective.py).
"""

import os
import io
import json
import argparse
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset, concatenate_datasets, Dataset


# System prompt for visual thinking (same as transform_perspective.py)
VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

# Dataset splits
SPLITS = [
    'distance_change_closer',
    'distance_change_further',
    'relative_position_left_left',
    'relative_position_left_right',
    'relative_position_right_left',
    'relative_position_right_right'
]

# Paths
BASE_DIR = Path("/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training")
JSON_DIR = BASE_DIR / "tmp_data"
OUTPUT_DIR = BASE_DIR / "data/training"

# Global CoT dictionaries (loaded once, used by map functions)
IMAGECOT = None
TEXTCOT = None


def load_json_cots():
    """Load CoT text from JSON files."""
    global IMAGECOT, TEXTCOT

    imagecot_path = JSON_DIR / "imagecot.json"
    textcot_path = JSON_DIR / "textcot.json"

    print(f"Loading imagecot.json...")
    with open(imagecot_path, 'r') as f:
        IMAGECOT = json.load(f)

    print(f"Loading textcot.json...")
    with open(textcot_path, 'r') as f:
        TEXTCOT = json.load(f)

    print(f"Loaded {len(IMAGECOT)} imagecot, {len(TEXTCOT)} textcot entries")


def format_question(item):
    """Format question with answer choices."""
    question = item['question'].strip()
    choices = item.get('answer_choices', [])

    if isinstance(choices, str):
        import ast
        try:
            choices = ast.literal_eval(choices)
        except:
            choices = [choices]

    if choices:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, choice in enumerate(choices):
            label = letters[i] if i < len(letters) else str(i)
            question += f"\n({label}) {choice}"

    return question


def transform_mmcot(item):
    """Transform function for MM CoT (used by datasets.map)."""
    # Build key from split_name and index
    split_name = item['_split_name']
    idx = item['_index']
    key = f"{split_name}{idx}"

    if key not in IMAGECOT:
        return None

    cot_text = IMAGECOT[key]
    answer = item['answer']

    # Split at <image> tag
    parts = cot_text.split('<image>')
    before_image = parts[0].strip()
    after_image = parts[1].strip() if len(parts) > 1 else ""

    # Get images as bytes
    marked_img = item['marked_image']
    new_persp_img = item['new_perspective']

    image_list = []
    for img in [marked_img, new_persp_img]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_list.append(buffer.getvalue())

    # Format instruction
    instruction = VLM_THINK_SYSTEM_PROMPT + format_question(item)

    return {
        "image_list": image_list,
        "instruction_list": [instruction],
        "output_text_list": [
            f"<think>{before_image}</think><image_start>",
            f"<image_end><think>{after_image}</think><answer>{answer}</answer>"
        ]
    }


def transform_textcot(item):
    """Transform function for Text CoT (used by datasets.map)."""
    # Build key from split_name and index
    split_name = item['_split_name']
    idx = item['_index']
    key = f"{split_name}{idx}"

    if key not in TEXTCOT:
        return None

    cot_text = TEXTCOT[key]
    answer = item['answer']

    # Get only input image as bytes
    marked_img = item['marked_image']
    if marked_img.mode != "RGB":
        marked_img = marked_img.convert("RGB")
    buffer = io.BytesIO()
    marked_img.save(buffer, format="PNG")
    image_list = [buffer.getvalue()]

    # Format instruction
    instruction = VLM_THINK_SYSTEM_PROMPT + format_question(item)

    return {
        "image_list": image_list,
        "instruction_list": [instruction],
        "output_text_list": [
            f"<think>{cot_text}</think><answer>{answer}</answer>"
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Create MM CoT and Text CoT datasets")
    parser.add_argument("--debug", action="store_true", help="Run on small subset (10 per split)")
    parser.add_argument("--mmcot-only", action="store_true", help="Only create MM CoT dataset")
    parser.add_argument("--textcot-only", action="store_true", help="Only create Text CoT dataset")
    parser.add_argument("--num-proc", type=int, default=64, help="Number of processes for map")
    args = parser.parse_args()

    # Load JSON CoT data
    load_json_cots()

    # Load and concatenate all splits
    print("\nLoading source dataset(s)...")
    all_dss = []

    for split in SPLITS:
        print(f"Loading split: {split}")

        if args.debug:
            ds = load_dataset(
                "weikaih/ai2thor-perspective-qa-20k-balanced-splits-with-obj",
                split=split,
                streaming=True
            )
            ds_list = list(ds.take(10))
            ds = Dataset.from_list(ds_list)
        else:
            ds = load_dataset(
                "weikaih/ai2thor-perspective-qa-20k-balanced-splits-with-obj",
                split=split
            )

        # Add split_name and index columns for key lookup
        ds = ds.add_column("_split_name", [split] * len(ds))
        ds = ds.add_column("_index", list(range(len(ds))))

        all_dss.append(ds)

    print("Concatenating datasets...")
    full_ds = concatenate_datasets(all_dss)

    if args.debug:
        print(f"Debug mode: {len(full_ds)} total items")
    else:
        print(f"Full mode: {len(full_ds)} total items")

    num_proc = 1 if args.debug else args.num_proc

    # Process MM CoT
    if not args.textcot_only:
        print(f"\nTransforming MM CoT dataset (num_proc={num_proc})...")
        mmcot_ds = full_ds.map(
            transform_mmcot,
            remove_columns=full_ds.column_names,
            num_proc=num_proc,
        )

        # Filter out None results
        mmcot_ds = mmcot_ds.filter(lambda x: x['image_list'] is not None)

        # Save
        dataset_name = "perspective_mmcot"
        data_dir = OUTPUT_DIR / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)

        num_shards = 1 if args.debug else 5
        metadata_info = {}
        total_samples = 0

        print(f"Saving to {num_shards} shards in {data_dir}...")
        for i in range(num_shards):
            shard = mmcot_ds.shard(num_shards=num_shards, index=i)
            filename = f"chunk_{i}.parquet"
            filepath = data_dir / filename

            shard.to_parquet(str(filepath))

            pf = pq.ParquetFile(filepath)
            metadata_info[str(filepath)] = {
                "num_row_groups": pf.num_row_groups,
                "num_rows": shard.num_rows
            }
            total_samples += shard.num_rows
            print(f"Saved {filename}: {shard.num_rows} rows")

        json_path = data_dir / "parquet_info.json"
        with open(json_path, 'w') as f:
            json.dump(metadata_info, f, indent=4)

        print(f"MM CoT total samples: {total_samples}")

    # Process Text CoT
    if not args.mmcot_only:
        print(f"\nTransforming Text CoT dataset (num_proc={num_proc})...")
        textcot_ds = full_ds.map(
            transform_textcot,
            remove_columns=full_ds.column_names,
            num_proc=num_proc,
        )

        # Filter out None results
        textcot_ds = textcot_ds.filter(lambda x: x['image_list'] is not None)

        # Save
        dataset_name = "perspective_textcot"
        data_dir = OUTPUT_DIR / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)

        num_shards = 1 if args.debug else 5
        metadata_info = {}
        total_samples = 0

        print(f"Saving to {num_shards} shards in {data_dir}...")
        for i in range(num_shards):
            shard = textcot_ds.shard(num_shards=num_shards, index=i)
            filename = f"chunk_{i}.parquet"
            filepath = data_dir / filename

            shard.to_parquet(str(filepath))

            pf = pq.ParquetFile(filepath)
            metadata_info[str(filepath)] = {
                "num_row_groups": pf.num_row_groups,
                "num_rows": shard.num_rows
            }
            total_samples += shard.num_rows
            print(f"Saved {filename}: {shard.num_rows} rows")

        json_path = data_dir / "parquet_info.json"
        with open(json_path, 'w') as f:
            json.dump(metadata_info, f, indent=4)

        print(f"Text CoT total samples: {total_samples}")

    print("\nDone!")


if __name__ == "__main__":
    main()
