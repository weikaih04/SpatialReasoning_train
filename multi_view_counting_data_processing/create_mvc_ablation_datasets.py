#!/usr/bin/env python3
"""
Create ablation datasets for Multi-View Counting (MVC).

This script creates:
1. mvc_no_thought: Direct answer without visual thinking
2. mvc_latent4: Visual thought with 64x64 output (4x4 latent, 16 tokens)
3. mvc_latent16: Visual thought with 256x256 output (16x16 latent, 64 tokens)
4. mvc_latent32: Visual thought with 512x512 output (32x32 latent, 256 tokens)

Input format (multi_view_counting):
{
    "image_list": [frame_0, frame_1, frame_2, frame_3, topdown_map],
    "instruction_list": [system_prompt + question],
    "output_text_list": [
        "<think>...</think><image_start>",
        "<image_end><answer>X</answer>"
    ],
    "num_input_images": 4
}

Output format (no_thought):
{
    "image_list": [frame_0, frame_1, frame_2, frame_3],  # No topdown map
    "instruction_list": [system_prompt + question],
    "output_text_list": ["<answer>X</answer>"],  # Direct answer
    "num_input_images": 4
}

Output format (latent variants):
Same as input, but output image is resized to target resolution.
"""

import os
import re
import json
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from pathlib import Path
from PIL import Image
import io

BASE_DIR = Path("/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training")
INPUT_DIR = BASE_DIR / "bagel_example/editing/multi_view_counting_training_v5"
OUTPUT_BASE = BASE_DIR / "bagel_example/editing"


def extract_answer(output_text_list: list) -> str:
    """Extract the answer from output_text_list."""
    if len(output_text_list) < 2:
        return None
    second_output = output_text_list[1]
    match = re.search(r'<answer>.*?</answer>', second_output, flags=re.DOTALL)
    if match:
        return match.group(0)
    return None


def resize_image(image_bytes: bytes, target_size: int) -> bytes:
    """Resize image to target size and return as bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def process_no_thought(input_path: Path, output_path: Path) -> tuple:
    """Create no-thought baseline: remove output image, direct answer only."""
    pf = pq.ParquetFile(input_path)
    processed_rows = []
    num_skipped = 0

    for batch in pf.iter_batches(batch_size=100):
        df = batch.to_pandas()
        for idx in range(len(df)):
            row = df.iloc[idx]
            answer = extract_answer(row['output_text_list'].tolist() if hasattr(row['output_text_list'], 'tolist') else row['output_text_list'])
            if answer is None:
                num_skipped += 1
                continue

            image_list = row['image_list'].tolist() if hasattr(row['image_list'], 'tolist') else row['image_list']
            num_input = row.get('num_input_images', 4)

            new_row = {
                'image_list': image_list[:num_input],  # Only input images
                'instruction_list': row['instruction_list'].tolist() if hasattr(row['instruction_list'], 'tolist') else row['instruction_list'],
                'output_text_list': [answer],  # Direct answer
                'num_input_images': num_input,
            }
            processed_rows.append(new_row)

    new_df = pd.DataFrame(processed_rows)
    new_table = pa.Table.from_pandas(new_df)
    pq.write_table(new_table, output_path)
    return len(processed_rows), num_skipped


def process_latent_variant(input_path: Path, output_path: Path, target_size: int) -> tuple:
    """Create latent variant with resized output image."""
    pf = pq.ParquetFile(input_path)
    processed_rows = []
    num_skipped = 0

    for batch in pf.iter_batches(batch_size=100):
        df = batch.to_pandas()
        for idx in range(len(df)):
            row = df.iloc[idx]

            image_list = row['image_list'].tolist() if hasattr(row['image_list'], 'tolist') else row['image_list']
            num_input = row.get('num_input_images', 4)

            if len(image_list) <= num_input:
                num_skipped += 1
                continue

            try:
                # Resize output image (topdown map)
                resized_output = resize_image(image_list[num_input], target_size)
                new_image_list = list(image_list[:num_input]) + [resized_output]

                new_row = {
                    'image_list': new_image_list,
                    'instruction_list': row['instruction_list'].tolist() if hasattr(row['instruction_list'], 'tolist') else row['instruction_list'],
                    'output_text_list': row['output_text_list'].tolist() if hasattr(row['output_text_list'], 'tolist') else row['output_text_list'],
                    'num_input_images': num_input,
                }
                processed_rows.append(new_row)
            except Exception as e:
                num_skipped += 1
                continue

    new_df = pd.DataFrame(processed_rows)
    new_table = pa.Table.from_pandas(new_df)
    pq.write_table(new_table, output_path)
    return len(processed_rows), num_skipped


def generate_parquet_info(output_dir: Path, total_samples: int, num_files: int):
    """Generate parquet_info.json for the dataset."""
    parquet_info = {
        "total_samples": total_samples,
        "num_files": num_files,
        "files": []
    }

    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.parquet'):
            filepath = output_dir / filename
            pf = pq.ParquetFile(filepath)
            parquet_info["files"].append({
                "filename": filename,
                "num_samples": pf.metadata.num_rows
            })

    info_path = output_dir / "parquet_info.json"
    with open(info_path, 'w') as f:
        json.dump(parquet_info, f, indent=2)
    print(f"Generated parquet_info.json at {info_path}")


def process_dataset(variant_name: str, process_func, **kwargs):
    """Process all parquet files for a variant."""
    output_dir = OUTPUT_BASE / f"multi_view_counting_{variant_name}"

    print(f"\n{'='*60}")
    print(f"Creating: {variant_name}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_skipped = 0
    num_files = 0

    parquet_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')])

    for filename in parquet_files:
        input_path = INPUT_DIR / filename
        output_path = output_dir / filename
        num_processed, num_skipped = process_func(input_path, output_path, **kwargs)
        total_samples += num_processed
        total_skipped += num_skipped
        num_files += 1
        print(f"  {filename}: {num_processed} samples ({num_skipped} skipped)")

    print(f"Total: {total_samples} samples in {num_files} files")
    generate_parquet_info(output_dir, total_samples, num_files)

    return total_samples


def main():
    print("Creating MVC Ablation Datasets")
    print(f"Input: {INPUT_DIR}")

    # 1. No-thought baseline
    process_dataset("no_thought", process_no_thought)

    # 2. Latent 4x4 (64x64 output)
    process_dataset("latent4", process_latent_variant, target_size=64)

    # 3. Latent 16x16 (256x256 output)
    process_dataset("latent16", process_latent_variant, target_size=256)

    # 4. Latent 32x32 (512x512 output)
    process_dataset("latent32", process_latent_variant, target_size=512)

    print("\n" + "="*60)
    print("All datasets created!")
    print("="*60)


if __name__ == "__main__":
    main()
