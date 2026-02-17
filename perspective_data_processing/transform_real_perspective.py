import os
import io
from datasets import load_dataset
import pyarrow.parquet as pq
import json
import argparse


VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


def transform_item(item):
    img0 = item['image_0']
    img1 = item['image_1']

    raw_images = [img0, img1]
    image_list = []

    for img in raw_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        b_io = io.BytesIO()
        img.save(b_io, format="PNG")
        image_list.append(b_io.getvalue())

    question = item['question'].strip()

    instruction_list = [VLM_THINK_SYSTEM_PROMPT + question]

    part1 = "<think>Let me visualize the new perspective.</think><image_start>"
    part2 = "<image_end><answer>I've generated the scene from the new perspective.</answer>"

    output_text_list = [part1, part2]

    return {
        "image_list": image_list,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run on small subset")
    args = parser.parse_args()

    print("Loading real_perspective_taking dataset...")
    if args.debug:
        from datasets import Dataset
        ds = load_dataset("MahtabBg/real_perspective_taking", split="train", streaming=True)
        ds_list = list(ds.take(20))
        ds = Dataset.from_list(ds_list)
    else:
        ds = load_dataset("MahtabBg/real_perspective_taking", split="train", num_proc=8)

    print(f"Loaded {len(ds)} items")

    print("Transforming dataset...")
    new_ds = ds.map(
        transform_item,
        remove_columns=ds.column_names,
        num_proc=64 if not args.debug else 1,
        features=None
    )

    dataset_name = "real_perspective"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "training", dataset_name)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    num_shards = 5 if not args.debug else 1
    metadata_info = {}
    total_samples = 0

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
        total_samples += shard.num_rows
        print(f"Saved {filename}: {shard.num_rows} rows")

    json_path = os.path.join(data_dir, "parquet_info.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_info, f, indent=4)

    print(f"Saved metadata to {json_path}")
    print(f"Total samples: {total_samples}")

    if args.debug:
        print("\n[DEBUG] Verifying first transformed item:")
        print(f"instruction: {new_ds[0]['instruction_list'][0][:200]}")
        print(f"output[0]: {new_ds[0]['output_text_list'][0]}")
        print(f"output[1]: {new_ds[0]['output_text_list'][1]}")
        print(f"num_images: {len(new_ds[0]['image_list'])}")


if __name__ == "__main__":
    main()
