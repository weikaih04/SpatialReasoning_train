"""
Transform multi-view counting dataset to unified_edit parquet format.

This script converts weikaih/multi_view_counting_training_v5 to the ThinkMorph
unified format with:
- image_list: [frame_0, frame_1, frame_2, frame_3, topdown_map]
- num_input_images: 4 (first 4 images are inputs, last 1 is output)
- instruction_list: [system_prompt + question]
- output_text_list: [think_part, answer_part]

Usage:
    python multi_view_counting_data_processing/transform_multi_view_counting.py --debug
    python multi_view_counting_data_processing/transform_multi_view_counting.py
"""

import os
import io
import json
import argparse
from datasets import load_dataset
import pyarrow.parquet as pq


VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


def transform_item(item):
    """Transform a single multi-view counting item to unified format.

    Input images (conditioning, no loss):
        - frame_0, frame_1, frame_2, frame_3: First-person view frames

    Output image (with loss):
        - topdown_map: Top-down view for reasoning

    The num_input_images field tells the dataloader how many images
    in image_list are inputs (first 4) vs outputs (last 1).
    """
    # Get frames and topdown map
    frames = [
        item.get('frame_0'),
        item.get('frame_1'),
        item.get('frame_2'),
        item.get('frame_3'),
    ]
    topdown = item.get('topdown_map')

    # Skip if any required image is missing
    if any(f is None for f in frames) or topdown is None:
        return {
            "image_list": None,
            "instruction_list": None,
            "output_text_list": None,
            "num_input_images": None,
        }

    # Convert images to bytes: [frame_0, frame_1, frame_2, frame_3, topdown_map]
    image_list = []
    for img in frames + [topdown]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        b_io = io.BytesIO()
        img.save(b_io, format="PNG")
        image_list.append(b_io.getvalue())

    # Get question (already contains choices in this dataset)
    question = item.get('question', '')
    if not question:
        return {
            "image_list": None,
            "instruction_list": None,
            "output_text_list": None,
            "num_input_images": None,
        }

    # Build instruction with system prompt
    instruction_list = [VLM_THINK_SYSTEM_PROMPT + question]

    # Get answer
    answer = item.get('answer', '')
    if not answer:
        return {
            "image_list": None,
            "instruction_list": None,
            "output_text_list": None,
            "num_input_images": None,
        }

    # Build output text
    output_text_list = [
        "<think>Use the top-down map to reason.</think><image_start>",
        f"<image_end><answer>{answer}</answer>"
    ]

    return {
        "image_list": image_list,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list,
        "num_input_images": 4,  # First 4 images are inputs, last 1 is output
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transform multi-view counting dataset to unified parquet format"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on small subset (100 items) for testing"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="weikaih/multi_view_counting_training_v5",
        help="HuggingFace dataset path"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=32,
        help="Number of processes for parallel processing"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=5,
        help="Number of output shards"
    )
    args = parser.parse_args()

    print(f"Loading source dataset: {args.dataset_name}...")

    if args.debug:
        # Use streaming to avoid full download in debug mode
        ds = load_dataset(args.dataset_name, split="train", streaming=True)
        ds_list = list(ds.take(100))
        from datasets import Dataset
        ds = Dataset.from_list(ds_list)
        print(f"Debug mode: {len(ds)} items")
    else:
        ds = load_dataset(args.dataset_name, split="train")
        print(f"Full mode: {len(ds)} items")

    num_proc = 1 if args.debug else args.num_proc
    print(f"Transforming dataset with {num_proc} processes...")

    # Use writer_batch_size to avoid PyArrow overflow when concatenating large arrays
    new_ds = ds.map(
        transform_item,
        remove_columns=ds.column_names,
        num_proc=num_proc,
        writer_batch_size=100,  # Write in smaller batches to avoid overflow
    )

    # Filter out None results
    original_len = len(new_ds)
    new_ds = new_ds.filter(lambda x: x.get('image_list') is not None)
    filtered_len = len(new_ds)
    print(f"Filtered: {original_len} -> {filtered_len} samples ({original_len - filtered_len} skipped)")

    # Define output structure
    dataset_name = args.dataset_name.rstrip('/').split('/')[-1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "bagel_example", "editing")

    data_dir = os.path.join(base_dir, dataset_name)
    info_dir = os.path.join(base_dir, "parquet_info")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)

    # Shard and save
    num_shards = 1 if args.debug else args.num_shards
    metadata_info = {}

    print(f"Saving to {num_shards} shards in {data_dir}...")
    for i in range(num_shards):
        shard = new_ds.shard(num_shards=num_shards, index=i)
        filename = f"chunk_{i}.parquet"
        filepath = os.path.join(data_dir, filename)

        shard.to_parquet(filepath)

        pf = pq.ParquetFile(filepath)
        metadata_info[filepath] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": shard.num_rows
        }
        print(f"Saved {filename}: {shard.num_rows} rows")

    total_rows = sum(info["num_rows"] for info in metadata_info.values())

    # Save metadata json
    json_path = os.path.join(info_dir, f"{dataset_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_info, f, indent=4)

    print(f"\nSaved metadata to {json_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"Total samples: {total_rows}")
    print(f"Data directory: {data_dir}")
    print(f"Parquet info: {json_path}")
    print(f"{'=' * 60}")

    # Print suggested dataset_info.py entry
    print("\nSuggested dataset_info.py entry:")
    print(f"""
'multi_view_counting': {{
    'data_dir': '{data_dir}',
    'num_files': {num_shards},
    'num_total_samples': {total_rows},
    'parquet_info_path': '{json_path}',
}},
""")


if __name__ == "__main__":
    main()
