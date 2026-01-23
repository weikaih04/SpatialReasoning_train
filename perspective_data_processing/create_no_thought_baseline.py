#!/usr/bin/env python3
"""
Create No-Thought Baseline Dataset for Perspective Taking Ablation.

This script processes the perspective_with_sysprompt dataset to create
a baseline that directly answers the question WITHOUT any intermediate
thought or image generation.

Input format (perspective_with_sysprompt):
{
    "image_list": [input_image, output_image],
    "instruction_list": [system_prompt + question],
    "output_text_list": [
        "<think>reasoning...</think><image_start>",
        "<image_end><answer>A</answer>"
    ]
}

Output format (perspective_no_thought):
{
    "image_list": [input_image],  # Only input image, NO output image
    "instruction_list": [system_prompt + question],  # Keep same
    "output_text_list": ["<answer>A</answer>"]  # Direct answer only
}

This baseline tests whether intermediate image visualization helps.
"""

import os
import re
import json
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

# Paths
BASE_DIR = Path("/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training")
INPUT_DIR = BASE_DIR / "data/training/perspective_with_sysprompt"
OUTPUT_DIR = BASE_DIR / "data/training/perspective_no_thought"


def extract_answer(output_text_list: list) -> str:
    """
    Extract the answer from output_text_list.

    The answer is in output_text_list[1] in format: <image_end><answer>X</answer>
    We extract just <answer>X</answer>
    """
    if len(output_text_list) < 2:
        return None

    second_output = output_text_list[1]
    # Extract <answer>...</answer>
    match = re.search(r'<answer>.*?</answer>', second_output, flags=re.DOTALL)
    if match:
        return match.group(0)
    return None


def process_parquet(input_path: Path, output_path: Path) -> tuple:
    """
    Process a single parquet file to create no-thought baseline.

    Returns (num_processed, num_skipped)
    """
    pf = pq.ParquetFile(input_path)

    processed_rows = []
    num_skipped = 0

    for batch in pf.iter_batches(batch_size=100):
        df = batch.to_pandas()

        for idx in range(len(df)):
            row = df.iloc[idx]

            # Extract answer
            answer = extract_answer(row['output_text_list'])
            if answer is None:
                num_skipped += 1
                continue

            # Create new row with only input image and direct answer
            new_row = {
                'image_list': [row['image_list'][0]],  # Only input image
                'instruction_list': row['instruction_list'],  # Keep same
                'output_text_list': [answer],  # Direct answer only
            }
            processed_rows.append(new_row)

    # Convert to DataFrame and save
    import pandas as pd
    new_df = pd.DataFrame(processed_rows)
    new_table = pa.Table.from_pandas(new_df)
    pq.write_table(new_table, output_path)

    return len(processed_rows), num_skipped


def generate_parquet_info(output_dir: Path, total_samples: int, num_files: int):
    """Generate parquet_info.json for the new dataset."""
    parquet_info = {
        "total_samples": total_samples,
        "num_files": num_files,
        "files": []
    }

    # List all parquet files and get their sample counts
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.parquet'):
            filepath = output_dir / filename
            pf = pq.ParquetFile(filepath)
            parquet_info["files"].append({
                "filename": filename,
                "num_samples": pf.metadata.num_rows
            })

    # Save parquet_info.json
    info_path = output_dir / "parquet_info.json"
    with open(info_path, 'w') as f:
        json.dump(parquet_info, f, indent=2)

    print(f"Generated parquet_info.json at {info_path}")
    return info_path


def verify_transformation(output_dir: Path, num_samples: int = 3):
    """Verify the transformation by printing sample outputs."""
    print("\n" + "="*60)
    print("Verification: Sample outputs from processed dataset")
    print("="*60)

    parquet_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.parquet')])
    if not parquet_files:
        print("No parquet files found!")
        return

    first_file = output_dir / parquet_files[0]
    pf = pq.ParquetFile(first_file)

    count = 0
    for batch in pf.iter_batches(batch_size=1):
        df = batch.to_pandas()
        row = df.iloc[0]

        print(f"\nSample {count + 1}:")
        print(f"  image_list length: {len(row['image_list'])}")
        print(f"  instruction_list: {row['instruction_list'][0][:100]}...")
        print(f"  output_text_list: {row['output_text_list']}")

        count += 1
        if count >= num_samples:
            break


def main():
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Creating NO-THOUGHT baseline:")
    print("  - Remove output/intermediate image")
    print("  - Remove think tags and image generation")
    print("  - Keep only direct answer")
    print()

    # Remove old output directory if exists
    import shutil
    if OUTPUT_DIR.exists():
        print(f"Removing existing output directory...")
        shutil.rmtree(OUTPUT_DIR)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all parquet files
    total_samples = 0
    total_skipped = 0
    num_files = 0

    parquet_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')])
    print(f"Found {len(parquet_files)} parquet files to process")

    for filename in parquet_files:
        input_path = INPUT_DIR / filename
        output_path = OUTPUT_DIR / filename

        num_processed, num_skipped = process_parquet(input_path, output_path)
        total_samples += num_processed
        total_skipped += num_skipped
        num_files += 1
        print(f"Processed: {filename} ({num_processed} samples, {num_skipped} skipped)")

    print(f"\nTotal: {total_samples} samples in {num_files} files ({total_skipped} skipped)")

    # Generate parquet_info.json
    generate_parquet_info(OUTPUT_DIR, total_samples, num_files)

    # Verify transformation
    verify_transformation(OUTPUT_DIR)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
