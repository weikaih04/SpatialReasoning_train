#!/usr/bin/env python3
"""Convert weikaih/habitat-perspective-qa-train-v2 HuggingFace dataset to parquet format for training."""

import os
import io
import json
import argparse
from datasets import load_dataset, concatenate_datasets
import pyarrow.parquet as pq

VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

THINK_TEMPLATES = {
    'perspective_distance_change': (
        "To determine if the {object} is closer or further, "
        "I need to visualize the new perspective after moving to the "
        "target position and turning {direction}."
    ),
    'perspective_relative_position': (
        "To determine if the {object} will be on the left or right, "
        "I need to visualize the new perspective after moving to the "
        "target position and turning {direction}."
    ),
}


def pil_to_bytes(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def transform_item(item):
    # Images: input (marked_image_no_arrow) + output (new_perspective)
    input_img = item['marked_image_no_arrow']
    output_img = item['new_perspective']
    image_list = [pil_to_bytes(input_img), pil_to_bytes(output_img)]

    # Question with choices
    question = item['question_no_arrow']
    choices = item['answer_choices']
    if isinstance(choices, str):
        choices = json.loads(choices)

    choices_str = ""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, choice in enumerate(choices):
        label = letters[i] if i < len(letters) else str(i)
        choices_str += f"\n({label}) {choice}"

    instruction_list = [VLM_THINK_SYSTEM_PROMPT + question + choices_str]

    # Map text answer to letter
    answer_text = str(item['answer']).strip().lower()
    answer_letter = None
    for i, choice in enumerate(choices):
        if str(choice).strip().lower() == answer_text:
            answer_letter = letters[i]
            break
    if answer_letter is None:
        # Fallback: first letter match
        answer_letter = "A"

    # Think text
    q_type = item.get('question_type', 'perspective_distance_change')
    obj = item.get('object_query', 'object')
    direction = item.get('turn_direction', 'specified direction')
    template = THINK_TEMPLATES.get(q_type, THINK_TEMPLATES['perspective_distance_change'])
    think_text = template.format(object=obj, direction=direction)

    part1 = f"<think>{think_text}</think><image_start>"
    part2 = f"<image_end><answer>{answer_letter}</answer>"
    output_text_list = [part1, part2]

    return {
        "image_list": image_list,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/data/training/habitat_perspective_v2")
    parser.add_argument("--num_shards", type=int, default=5)
    args = parser.parse_args()

    print("Loading weikaih/habitat-perspective-qa-train-v2...")
    ds_dict = load_dataset("weikaih/habitat-perspective-qa-train-v2")

    # Concatenate all splits
    all_splits = []
    for split_name in sorted(ds_dict.keys()):
        print(f"  Split {split_name}: {len(ds_dict[split_name])} samples")
        all_splits.append(ds_dict[split_name])
    ds = concatenate_datasets(all_splits)
    print(f"Total: {len(ds)} samples")

    print("Transforming dataset...")
    new_ds = ds.map(
        transform_item,
        remove_columns=ds.column_names,
        num_proc=8,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Saving to {args.num_shards} shards in {args.output_dir}...")
    metadata_info = {}
    for i in range(args.num_shards):
        shard = new_ds.shard(num_shards=args.num_shards, index=i)
        filename = f"chunk_{i}.parquet"
        filepath = os.path.join(args.output_dir, filename)
        shard.to_parquet(filepath)

        pf = pq.ParquetFile(filepath)
        metadata_info[filepath] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": shard.num_rows,
        }
        print(f"  {filename}: {shard.num_rows} rows")

    # Save parquet_info.json
    info_path = os.path.join(args.output_dir, "parquet_info.json")
    with open(info_path, "w") as f:
        json.dump(metadata_info, f, indent=4)
    print(f"Saved metadata to {info_path}")

    # Verify
    print("\nVerifying first sample of chunk_0:")
    pf = pq.ParquetFile(os.path.join(args.output_dir, "chunk_0.parquet"))
    t = pf.read_row_group(0)
    row = t.to_pydict()
    print(f"  image_list count: {len(row['image_list'][0])}")
    print(f"  instruction: {row['instruction_list'][0][0][:150]}...")
    print(f"  output_text: {row['output_text_list'][0]}")
    print("\nDone!")


if __name__ == "__main__":
    main()
