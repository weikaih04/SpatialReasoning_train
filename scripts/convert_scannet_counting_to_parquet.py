#!/usr/bin/env python3
"""
Convert leo66666/scannet_counting (train split) to ThinkMorph unified_edit parquet format.

Visual thought only (no text reasoning): input images + reasoning_image_0 as visual thought + answer.
Format matches messytable (single output image with empty think tags).

Output schema per row:
  - image_list: [view_0_bytes, ..., view_N_bytes, reasoning_image_0_bytes]
  - instruction_list: [system_prompt + question + MCQ choices]
  - output_text_list: ["<think></think><image_start>", "<image_end><answer>X</answer>"]
  - num_input_images: N  (5-8 input views, 1 output reasoning image)
"""

import argparse
import io
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import snapshot_download

VLM_THINK_SYSTEM_PROMPT = (
    "\nLet's think step by step to answer the question. For text-based thinking, "
    "enclose the process within <think> </think>, e.g. <think> thinking process "
    "here </think>. For visual thinking, enclose the content within <image_start> "
    "</image_end>, e.g. <image_start> thinking image here </image_end>. Finally "
    "conclude with the final answer wrapped in <answer></answer> tags, "
    "i.e.<answer> answer here </answer>.\n"
)

LETTERS = "ABCD"


def generate_mcq_choices(gt_answer, seed):
    """Generate 4 MCQ choices (A/B/C/D) with gt_answer included and 3 distractors."""
    rng = random.Random(seed)
    pool = [x for x in range(max(1, gt_answer - 3), min(gt_answer + 4, gt_answer + 3 + 1)) if x != gt_answer]
    if len(pool) < 3:
        pool = [x for x in range(max(1, gt_answer - 5), gt_answer + 6) if x != gt_answer]
    distractors = rng.sample(pool, min(3, len(pool)))
    # Fallback if still not enough
    while len(distractors) < 3:
        extra = gt_answer + len(distractors) + 1
        if extra != gt_answer and extra not in distractors:
            distractors.append(extra)
    options = distractors + [gt_answer]
    rng.shuffle(options)
    correct_letter = LETTERS[options.index(gt_answer)]
    return options, correct_letter


def image_to_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def reformat_sample(sample, idx):
    gt_answer_str = sample["gt_answer"]
    question = sample["question"].strip()
    images = sample["images"]
    reasoning_img = sample.get("reasoning_image_0")

    if not images or gt_answer_str is None or not question or reasoning_img is None:
        return None

    try:
        gt_answer = int(gt_answer_str)
    except (ValueError, TypeError):
        return None

    options, correct_letter = generate_mcq_choices(gt_answer, seed=42 + idx)

    choices_str = "\n".join(f"({LETTERS[i]}) {v}" for i, v in enumerate(options))
    full_question = f"{question}\n{choices_str}"

    # Input images (5-8 views) + 1 output image (reasoning_image_0)
    image_bytes = [image_to_bytes(img) for img in images]
    image_bytes.append(image_to_bytes(reasoning_img))

    return {
        "image_list": image_bytes,
        "instruction_list": [VLM_THINK_SYSTEM_PROMPT + full_question],
        "output_text_list": [
            "<think></think><image_start>",
            f"<image_end><answer>{correct_letter}</answer>",
        ],
        "num_input_images": len(images),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/"
                "ThinkMorph_training/bagel_example/editing/scannet_counting",
    )
    parser.add_argument("--num_shards", type=int, default=5)
    args = parser.parse_args()

    # Download dataset files first with multi-threaded download
    print("Downloading leo66666/scannet_counting with multi-threaded download...")
    local_path = snapshot_download(
        "leo66666/scannet_counting",
        repo_type="dataset",
        max_workers=8,
    )
    print(f"Downloaded to: {local_path}")

    print("Loading dataset from local files...")
    ds = load_dataset(local_path, split="train")
    print(f"Total train samples: {len(ds)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Convert all samples
    converted = []
    skipped = 0
    for idx, sample in enumerate(ds):
        row = reformat_sample(sample, idx)
        if row is None:
            skipped += 1
            continue
        converted.append(row)
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(ds)}...")

    print(f"Converted: {len(converted)}, Skipped: {skipped}")

    # Shard and save as parquet
    metadata_info = {}
    shard_size = len(converted) // args.num_shards
    for i in range(args.num_shards):
        start = i * shard_size
        end = start + shard_size if i < args.num_shards - 1 else len(converted)
        shard = converted[start:end]

        # Build PyArrow table
        table = pa.table(
            {
                "image_list": pa.array([row["image_list"] for row in shard],
                                       type=pa.list_(pa.binary())),
                "instruction_list": pa.array([row["instruction_list"] for row in shard],
                                             type=pa.list_(pa.string())),
                "output_text_list": pa.array([row["output_text_list"] for row in shard],
                                             type=pa.list_(pa.string())),
                "num_input_images": pa.array([row["num_input_images"] for row in shard],
                                             type=pa.int64()),
            }
        )

        filepath = os.path.join(args.output_dir, f"chunk_{i}.parquet")
        pq.write_table(table, filepath, row_group_size=30)

        pf = pq.ParquetFile(filepath)
        metadata_info[filepath] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": len(shard),
        }
        print(f"  chunk_{i}.parquet: {len(shard)} rows")

    # Save parquet_info.json
    info_path = os.path.join(args.output_dir, "parquet_info.json")
    with open(info_path, "w") as f:
        json.dump(metadata_info, f, indent=4)
    print(f"Saved metadata to {info_path}")

    # Verify first sample
    print("\nVerifying first sample of chunk_0:")
    pf = pq.ParquetFile(os.path.join(args.output_dir, "chunk_0.parquet"))
    t = pf.read_row_group(0)
    row = t.to_pydict()
    print(f"  num_input_images: {row['num_input_images'][0]}")
    print(f"  image_list count: {len(row['image_list'][0])}")
    print(f"  instruction: {row['instruction_list'][0][0][:120]}...")
    print(f"  output_text: {row['output_text_list'][0]}")
    print("\nDone!")


if __name__ == "__main__":
    main()
