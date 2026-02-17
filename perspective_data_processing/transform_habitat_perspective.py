import os
import io
from datasets import load_dataset, concatenate_datasets
import pyarrow.parquet as pq
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import re

VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


def get_reasoning_template(question_type, question_text):
    direction = "specified direction"

    match_dir_1 = re.search(r"turn[s|ing]*\s+.*?(left|right)", question_text, re.IGNORECASE)
    match_dir_2 = re.search(r"(left|right)\s+turn", question_text, re.IGNORECASE)

    if match_dir_1:
         direction = match_dir_1.group(1).lower()
    elif match_dir_2:
         direction = match_dir_2.group(1).lower()

    parts = re.split(r'[.?!]', question_text)
    parts = [p.strip() for p in parts if p.strip()]
    last_sentence = parts[-1] if parts else question_text

    obj = "target object"

    match_obj = re.search(r"(?:would|will|find)\s+(?:the|a)\s+(?P<obj>\w+(?:\s+\w+){0,3}?)\s+(?:be|on)", last_sentence, re.IGNORECASE)

    if match_obj:
        obj = match_obj.group("obj")
    else:
        excluded = ["ground", "position", "direction", "same", "scene", "x"]
        candidates = re.findall(r"\bthe\s+(\w+)", last_sentence, re.IGNORECASE)
        for c in candidates:
            if c.lower() not in excluded:
                obj = c
                break

    if "distance" in question_type.lower():
        return f"To determine if the {obj} is closer or further, I need to visualize the new perspective after moving to the target position and turning {direction}."
    elif "relative" in question_type.lower() or "position" in question_type.lower():
        return f"To determine the {obj}'s relative position, I need to visualize the new perspective after moving to the target position and turning {direction}."
    else:
        return f"To answer this question about the {obj}, I need to visualize the new perspective after moving to the target position and turning {direction}."

def transform_item(item):
    marked_img = item['marked_image']
    new_persp_img = item['new_perspective']

    raw_images = [marked_img, new_persp_img]
    image_list = []

    for img in raw_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        b_io = io.BytesIO()
        img.save(b_io, format="PNG")
        image_list.append(b_io.getvalue())

    clean_question = item['question'].strip()

    choices = item.get('answer_choices', [])
    if isinstance(choices, str):
        import ast
        try:
            choices = ast.literal_eval(choices)
        except:
            choices = [choices]

    if choices:
        choices_str = ""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, choice in enumerate(choices):
            label = letters[i] if i < len(letters) else str(i)
            choices_str += f"\n({label}) {choice}"
        clean_question += choices_str

    instruction_list = [VLM_THINK_SYSTEM_PROMPT + clean_question]

    q_type = item.get('question_type', 'unknown')
    raw_question = item['question'].strip()
    thought_content = get_reasoning_template(q_type, raw_question)

    # Convert text answer to letter (A/B) to match AI2Thor format
    raw_answer = item.get('answer', '')
    if choices and raw_answer in choices:
        answer_idx = choices.index(raw_answer)
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        answer_text = letters[answer_idx] if answer_idx < len(letters) else raw_answer
    else:
        answer_text = raw_answer

    part1 = f"<think>{thought_content}</think><image_start>"
    part2 = f"<image_end><answer>{answer_text}</answer>"

    output_text_list = [part1, part2]

    return {
        "image_list": image_list,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list
    }

def load_split(split_name, debug=False):
    """Load a single split - designed to be called in parallel."""
    print(f"[Thread] Loading split: {split_name}")
    if debug:
        from datasets import Dataset
        ds = load_dataset("weikaih/habitat-perspective-qa-train", split=split_name, streaming=True)
        ds_list = list(ds.take(10))
        ds = Dataset.from_list(ds_list)
    else:
        ds = load_dataset("weikaih/habitat-perspective-qa-train", split=split_name, num_proc=8)
    print(f"[Thread] Finished split: {split_name} ({len(ds)} rows)")
    return split_name, ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run on small subset")
    args = parser.parse_args()

    print("Loading habitat perspective dataset (multithreaded)...")
    splits = [
        'distance_closer',
        'distance_further',
        'position_left_left',
        'position_left_right',
        'position_right_left',
        'position_right_right'
    ]

    # Download all 6 splits in parallel
    all_dss = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(load_split, s, args.debug): s for s in splits}
        for future in as_completed(futures):
            split_name, ds = future.result()
            all_dss[split_name] = ds

    # Concatenate in original order
    ordered_dss = [all_dss[s] for s in splits]

    print("Concatenating datasets...")
    full_ds = concatenate_datasets(ordered_dss)

    if args.debug:
        print(f"Debug mode: {len(full_ds)} total items")
    else:
        print(f"Full mode: {len(full_ds)} total items")

    print("Transforming dataset...")
    new_ds = full_ds.map(
        transform_item,
        remove_columns=full_ds.column_names,
        num_proc=64 if not args.debug else 1,
        features=None
    )

    dataset_name = "habitat_perspective"

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
        print(new_ds[0]['output_text_list'][0])
        print(new_ds[0]['output_text_list'][1])

if __name__ == "__main__":
    main()
