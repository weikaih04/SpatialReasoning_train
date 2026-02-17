#!/usr/bin/env python3
"""
Create MM CoT and Text CoT Datasets for Multi-View Counting (MVC).

Uses datasets.map() with num_proc for parallel processing.

MM CoT: Model reasons in text, generates topdown map, reasons more, then answers.
Text CoT: Model reasons only in text (frame-by-frame), no image generation.
"""

import os
import io
import json
import argparse
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset, Dataset


# System prompt for visual thinking
VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

# Paths
BASE_DIR = Path("/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training")
MMCOT_JSON_PATH = BASE_DIR / "mmcot_count_hfindex.json"
TEXTCOT_JSON_PATH = BASE_DIR / "textcot_count_hfindex.json"
OUTPUT_DIR = BASE_DIR / "bagel_example" / "editing"

# HuggingFace dataset
HF_DATASET = "weikaih/multi_view_counting_training_v5"

# Global CoT dictionaries (loaded once, used by map functions)
MMCOT = None
TEXTCOT = None


def load_json_cots():
    """Load CoT text from JSON files."""
    global MMCOT, TEXTCOT

    print(f"Loading mmcot_count_hfindex.json...")
    with open(MMCOT_JSON_PATH, 'r') as f:
        MMCOT = json.load(f)

    print(f"Loading textcot_count_hfindex.json...")
    with open(TEXTCOT_JSON_PATH, 'r') as f:
        TEXTCOT = json.load(f)

    print(f"Loaded {len(MMCOT)} mmcot, {len(TEXTCOT)} textcot entries")


def image_to_bytes(img):
    """Convert a PIL image to PNG bytes."""
    if img is None:
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def transform_mmcot(item):
    """Transform function for MM CoT (used by datasets.map).

    Output format:
    - image_list: [frame_0, frame_1, frame_2, frame_3, topdown_map] (4 input + 1 output)
    - num_input_images: 4
    - instruction_list: [system_prompt + question]
    - output_text_list: [
        "<think>{before_image}</think><image_start>",
        "<image_end><think>{after_image}</think><answer>{answer}</answer>"
      ]
    """
    key = str(item['_hf_index'])

    if key not in MMCOT:
        return {
            'image_list': None,
            'num_input_images': None,
            'instruction_list': None,
            'output_text_list': None,
        }

    cot_text = MMCOT[key]
    answer = item['answer']

    # Split at <image> tag (handle edge case: some entries use </image> instead)
    if '<image>' in cot_text:
        parts = cot_text.split('<image>')
    elif '</image>' in cot_text:
        parts = cot_text.split('</image>')
    else:
        # No image tag found, skip
        return {
            'image_list': None,
            'num_input_images': None,
            'instruction_list': None,
            'output_text_list': None,
        }

    before_image = parts[0].strip()
    after_image = parts[1].strip() if len(parts) > 1 else ""

    # Get images as bytes - need all 4 frames + topdown_map
    frame_imgs = [item.get('frame_0'), item.get('frame_1'), item.get('frame_2'), item.get('frame_3')]
    topdown_img = item.get('topdown_map')

    # Skip if any frame or topdown is missing
    if any(img is None for img in frame_imgs) or topdown_img is None:
        return {
            'image_list': None,
            'num_input_images': None,
            'instruction_list': None,
            'output_text_list': None,
        }

    image_list = [image_to_bytes(img) for img in frame_imgs] + [image_to_bytes(topdown_img)]

    # Check all conversions succeeded
    if any(b is None for b in image_list):
        return {
            'image_list': None,
            'num_input_images': None,
            'instruction_list': None,
            'output_text_list': None,
        }

    # Format instruction (question already contains choices in MVC dataset)
    instruction = VLM_THINK_SYSTEM_PROMPT + item['question']

    return {
        'image_list': image_list,
        'num_input_images': 4,
        'instruction_list': [instruction],
        'output_text_list': [
            f"<think>{before_image}</think><image_start>",
            f"<image_end><think>{after_image}</think><answer>{answer}</answer>"
        ]
    }


def transform_textcot(item):
    """Transform function for Text CoT (used by datasets.map).

    Output format:
    - image_list: [frame_0, frame_1, frame_2, frame_3] (4 input only, NO topdown)
    - num_input_images: 4
    - instruction_list: [system_prompt + question]
    - output_text_list: [
        "<think>{cot_text}</think><answer>{answer}</answer>"
      ]
    """
    key = str(item['_hf_index'])

    if key not in TEXTCOT:
        return {
            'image_list': None,
            'num_input_images': None,
            'instruction_list': None,
            'output_text_list': None,
        }

    cot_text = TEXTCOT[key]
    answer = item['answer']

    # Get images as bytes - only 4 frames, no topdown needed
    frame_imgs = [item.get('frame_0'), item.get('frame_1'), item.get('frame_2'), item.get('frame_3')]

    # Skip if any frame is missing
    if any(img is None for img in frame_imgs):
        return {
            'image_list': None,
            'num_input_images': None,
            'instruction_list': None,
            'output_text_list': None,
        }

    image_list = [image_to_bytes(img) for img in frame_imgs]

    # Check all conversions succeeded
    if any(b is None for b in image_list):
        return {
            'image_list': None,
            'num_input_images': None,
            'instruction_list': None,
            'output_text_list': None,
        }

    # Format instruction (question already contains choices in MVC dataset)
    instruction = VLM_THINK_SYSTEM_PROMPT + item['question']

    return {
        'image_list': image_list,
        'num_input_images': 4,
        'instruction_list': [instruction],
        'output_text_list': [
            f"<think>{cot_text}</think><answer>{answer}</answer>"
        ]
    }


def save_dataset(ds, dataset_name, num_shards, output_dir):
    """Save dataset to parquet shards and generate parquet_info.json."""
    data_dir = output_dir / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata_info = {}
    total_samples = 0

    print(f"Saving to {num_shards} shards in {data_dir}...")
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
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

    print(f"Total samples: {total_samples}")
    return total_samples


def main():
    parser = argparse.ArgumentParser(description="Create MM CoT and Text CoT datasets for MVC")
    parser.add_argument("--debug", action="store_true", help="Run on small subset (100 samples)")
    parser.add_argument("--mmcot-only", action="store_true", help="Only create MM CoT dataset")
    parser.add_argument("--textcot-only", action="store_true", help="Only create Text CoT dataset")
    parser.add_argument("--num-proc", type=int, default=64, help="Number of processes for map")
    args = parser.parse_args()

    # Load JSON CoT data
    load_json_cots()

    # Load HF dataset (single train split)
    print(f"\nLoading source dataset: {HF_DATASET}...")
    if args.debug:
        ds = load_dataset(HF_DATASET, split="train", streaming=True)
        ds_list = list(ds.take(100))
        full_ds = Dataset.from_list(ds_list)
    else:
        full_ds = load_dataset(HF_DATASET, split="train")

    # Add HF index column for key lookup
    full_ds = full_ds.add_column("_hf_index", list(range(len(full_ds))))

    print(f"Loaded {len(full_ds)} samples")

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

        num_shards = 1 if args.debug else 5
        total = save_dataset(mmcot_ds, "mvc_mmcot", num_shards, OUTPUT_DIR)
        print(f"\nMM CoT dataset saved: {total} samples")

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

        num_shards = 1 if args.debug else 5
        total = save_dataset(textcot_ds, "mvc_textcot", num_shards, OUTPUT_DIR)
        print(f"\nText CoT dataset saved: {total} samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
