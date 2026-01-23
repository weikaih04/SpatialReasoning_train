"""
Add system prompt to non-mmcot path tracing data.
The original data is already in bagel format but lacks system prompt.
"""
import os
import pyarrow.parquet as pq
import json
from datasets import Dataset, concatenate_datasets

VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

def transform_item(item):
    # Add system prompt to instruction_list
    original = item['instruction_list'][0]
    return {
        'image_list': item['image_list'],
        'instruction_list': [VLM_THINK_SYSTEM_PROMPT + original],
        'output_text_list': item['output_text_list']
    }

def main():
    # Source and destination paths
    src_dir = "/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/custom_datasets/path_tracing/bagel_example/editing/path-tracing-2point-balanced8-16k"
    dst_dir = "/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/path_tracing_with_sysprompt"

    os.makedirs(dst_dir, exist_ok=True)

    # Get all parquet files
    parquet_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.parquet')])
    print(f"Found {len(parquet_files)} parquet files")

    # Load all chunks and concatenate
    all_datasets = []
    for filename in parquet_files:
        src_path = os.path.join(src_dir, filename)
        print(f"Loading {filename}...")
        ds = Dataset.from_parquet(src_path)
        all_datasets.append(ds)

    print("Concatenating datasets...")
    full_ds = concatenate_datasets(all_datasets)
    print(f"Total samples: {len(full_ds)}")

    print("Transforming dataset...")
    new_ds = full_ds.map(
        transform_item,
        remove_columns=full_ds.column_names,
        num_proc=32,
        features=None
    )

    # Shard and save
    num_shards = 5
    metadata_info = {}
    total_samples = 0

    print(f"Saving to {num_shards} shards in {dst_dir}...")
    for i in range(num_shards):
        shard = new_ds.shard(num_shards=num_shards, index=i)
        filename = f"chunk_{i}.parquet"
        filepath = os.path.join(dst_dir, filename)

        shard.to_parquet(filepath)

        # Read back to get info
        pf = pq.ParquetFile(filepath)
        metadata_info[filepath] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": shard.num_rows
        }
        total_samples += shard.num_rows
        print(f"Saved {filename}: {shard.num_rows} rows")

    # Save metadata
    json_path = os.path.join(dst_dir, "parquet_info.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_info, f, indent=4)

    print(f"\nDone! Total samples: {total_samples}")
    print(f"Output directory: {dst_dir}")

    # Verify
    print("\n=== Verification ===")
    print("instruction_list[0][:400]:")
    print(new_ds[0]['instruction_list'][0][:400])

if __name__ == "__main__":
    main()
