import os
import io
from datasets import load_dataset, concatenate_datasets
import pyarrow.parquet as pq
import json
import argparse

import re

VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''


def get_reasoning_template(question_type, question_text):
    # Extract direction
    # Improved logic: Check "turn... direction" AND "direction turn"
    direction = "specified direction"
    
    # 1. "turn... (left/right)" (e.g. "turns right", "turn left", "turning... right")
    match_dir_1 = re.search(r"turn[s|ing]*\s+.*?(left|right)", question_text, re.IGNORECASE)
    # 2. "(left/right) turn" (e.g. "right turn")
    match_dir_2 = re.search(r"(left|right)\s+turn", question_text, re.IGNORECASE)
    
    if match_dir_1:
         direction = match_dir_1.group(1).lower()
    elif match_dir_2:
         direction = match_dir_2.group(1).lower()
        
    # Extract object
    # User suggestion: extract noun from the last sentence.
    parts = re.split(r'[.?!]', question_text)
    parts = [p.strip() for p in parts if p.strip()]
    last_sentence = parts[-1] if parts else question_text
    
    obj = "target object"
    
    # Validated regex from debug_regex.py
    # Matches: "would the doorway be", "Will the stool be", "find the chair on"
    match_obj = re.search(r"(?:would|will|find)\s+(?:the|a)\s+(?P<obj>\w+(?:\s+\w+){0,3}?)\s+(?:be|on)", last_sentence, re.IGNORECASE)
    
    if match_obj:
        obj = match_obj.group("obj")
    else:
        # Fallback: Find any "the [word]" that isn't a common setup word
        excluded = ["ground", "position", "direction", "same", "scene", "x"]
        candidates = re.findall(r"\bthe\s+(\w+)", last_sentence, re.IGNORECASE)
        for c in candidates:
            if c.lower() not in excluded:
                obj = c
                break

    # Based on implementation plan
    if "distance" in question_type.lower():
        return f"To determine if the {obj} is closer or further, I need to visualize the new perspective after moving to the target position and turning {direction}."
    elif "relative" in question_type.lower() or "position" in question_type.lower():
        return f"To determine the {obj}'s relative position, I need to visualize the new perspective after moving to the target position and turning {direction}."
    else:
        # Fallback
        return f"To answer this question about the {obj}, I need to visualize the new perspective after moving to the target position and turning {direction}."

def transform_item(item):
    # 1. Images
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

    # 2. Instruction
    clean_question = item['question'].strip()
    
    # Format answer choices
    choices = item.get('answer_choices', [])
    if isinstance(choices, str):
        import ast
        try:
            choices = ast.literal_eval(choices)
        except:
            choices = [choices] # Fallback
            
    if choices:
        choices_str = ""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, choice in enumerate(choices):
            # Check if answer is just the letter or matches index
            # The dataset 'answer' is 'A', so we should label choices with A, B, etc.
            label = letters[i] if i < len(letters) else str(i)
            choices_str += f"\n({label}) {choice}"
        clean_question += choices_str
    
    instruction_list = [VLM_THINK_SYSTEM_PROMPT + clean_question]
    
    # 3. Output Text (Reasoning)
    # Generate template
    q_type = item.get('question_type', 'unknown')
    # Use raw question for template extraction to avoid appended choices interfering with regex
    raw_question = item['question'].strip()
    thought_content = get_reasoning_template(q_type, raw_question)
    
    answer_text = item.get('answer', '') 
    # NOTE: The example showed 'answer': 'A'. 
    # We might want the text answer "Closer" or "Further"? 
    # But usually Bagel expects the direct answer string.
    # Let's inspect answer field in debug to be sure. 
    # If answer is 'A', we might need to map it to choices[0].
    # But for now, let's use what is provided.
    
    part1 = f"<think>{thought_content}</think><image_start>"
    part2 = f"<image_end><answer>{answer_text}</answer>"
    
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

    print("Loading source dataset(s)...")
    splits = [
        'distance_change_closer', 
        'distance_change_further', 
        'relative_position_left_left', 
        'relative_position_left_right', 
        'relative_position_right_left', 
        'relative_position_right_right'
    ]
    
    all_dss = []
    for s in splits:
        print(f"Loading split: {s}")
        if args.debug:
            # Use streaming to avoid full download
            ds = load_dataset("weikaih/ai2thor-perspective-qa-20k-balanced-splits", split=s, streaming=True)
            # Take only 10 items and convert to list/Dataset to allow concatenation with others efficiently for debug
            ds_list = list(ds.take(10))
            from datasets import Dataset
            ds = Dataset.from_list(ds_list)
        else:
            ds = load_dataset("weikaih/ai2thor-perspective-qa-20k-balanced-splits", split=s)
        all_dss.append(ds)
        
    print("Concatenating datasets...")
    full_ds = concatenate_datasets(all_dss)
    
    if args.debug:
        print(f"Debug mode: {len(full_ds)} total items")
    else:
        print(f"Full mode: {len(full_ds)} total items")
        
    print("Transforming dataset...")
    new_ds = full_ds.map(
        transform_item, 
        remove_columns=full_ds.column_names, 
        num_proc=4 if not args.debug else 1,
        features=None 
    )
    
    # Define output structure
    # Define output structure
    dataset_name = "perspective_qa"
    
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
    num_shards = 5 if not args.debug else 1
    metadata_info = {}
    
    print(f"Saving to {num_shards} shards in {data_dir}...")
    for i in range(num_shards):
        shard = new_ds.shard(num_shards=num_shards, index=i)
        filename = f"chunk_{i}.parquet"
        filepath = os.path.join(data_dir, filename)
        
        shard.to_parquet(filepath)
        
        # Read back to get info
        pf = pq.ParquetFile(filepath)
        # Store relative path or absolute? Previous verified used relative key in JSON but absolute in py usage?
        # Actually the example json had specific keys. Let's stick to filepath.
        metadata_info[filepath] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": shard.num_rows
        }
        print(f"Saved {filename}: {shard.num_rows} rows")

    # Save metadata json
    json_path = os.path.join(info_dir, f"{dataset_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_info, f, indent=4)
        
    print(f"Saved metadata to {json_path}")
    
    # If debug, print verification
    if args.debug:
        print("\n[DEBUG] Verifying first transformed item:")
        print(new_ds[0]['output_text_list'][0])
        print(new_ds[0]['output_text_list'][1])

if __name__ == "__main__":
    main()
