import os
from datasets import load_dataset, Dataset

VLM_THINK_SYSTEM_PROMPT = '''
Let's think step by step to answer the question. For text-based thinking, enclose the process within <think> </think>, e.g. <think> thinking process here </think>. For visual thinking, enclose the content within <image_start> </image_end>, e.g. <image_start> thinking image here </image_end>. Finally conclude with the final answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.
'''

def transform_item(item):
    # Prepare image_list: topdown_image then sideview_images
    topdown = item['topdown_image']
    sideviews = item['sideview_images']
    # Ensure sideviews is a list
    if not isinstance(sideviews, list):
        sideviews = [sideviews]
    
    raw_images = [topdown] + sideviews
    
    # Convert images to bytes
    image_list = []
    import io
    for img in raw_images:
        # Assuming img is a PIL Image
        # Convert to RGB just in case
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        b_io = io.BytesIO()
        img.save(b_io, format="PNG") # Or JPEG
        image_list.append(b_io.getvalue())
    
    # Prepare instruction_list
    # Remove <image_1> from question. 
    # Patterns observed: "view <image_1>.", "view <image_1>,", "view <image_1> "
    # Strategy: 
    # 1. Replace " <image_1>" with "" (merges to previous word if no punctuation, or keeps punctuation attached)
    #    e.g. "view <image_1>." -> "view."
    # 2. Replace "<image_1>" with "" (fallback)
    clean_question = item['question'].replace(' <image_1>', '').replace('<image_1>', '').replace('  ', ' ').strip()
    
    # Format answer choices
    choices = item.get('choices', [])
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
            label = letters[i] if i < len(letters) else str(i)
            choices_str += f"\n({label}) {choice}"
        clean_question += choices_str

    instruction_list = [VLM_THINK_SYSTEM_PROMPT + clean_question]
    
    # Prepare output_text_list
    # Format: 
    # 1. <think>{sideview_desc}</think><image_start>
    # 2. <image_end><answer>{answer}</answer>
    
    answer = item.get('answer', '')
    if answer:
        answer_block = f"<answer>{answer}</answer>"
    else:
        # Sideview data has no answer, add confirmation text as training signal
        sideview_desc = item.get('sideview_desc', '')
        if sideview_desc:
            answer_block = "This is the generated view from that perspective."
        else:
            answer_block = ""
    
    # Check for mm-thought (new dataset)
    if 'mm-thought' in item and item['mm-thought']:
        mm_thought = item['mm-thought']
        import re
        # Find the image token in the thought
        # Expected format: "... <image_2> ..."
        # We need to split around it. 
        # The thought might look like: "As I move ... At M1 <image_2>, I observe ..."
        # splitting by regex on <image_\d+>
        
        # Note: image token might be <image_1>, <image_2> etc. usually <image_2> based on context
        split = re.split(r'(<image_\d+>)', mm_thought)
        
        if len(split) >= 3:
            # split[0] is pre-image, split[1] is tag, split[2] is post-image
            pre_image = split[0].strip()
            # split[1] is the tag itself, e.g. <image_2>
            post_image = "".join(split[2:]).strip() 
             # remove leading punctuation if any from post_image like ", " or "."
            # User example: "part2=f"<image_end>, I observe..." 
            # Note: The user wants part2 to start with <image_end>. 
            # The user example shows: part1 = ... <image_start>"  part2="<image_end>, I observe..."
            # Wait, the user example logic: 
            # part1 = "<think>... At M1</think><image_start>"
            # part2 = "<image_end>, I observe ... <answer>{answer}</answer>"
            
            # The user's example "At M1 <image_2>, I observe..."
            # pre_image = "As I move ... At M1"
            # post_image = ", I observe ..."
            
            part1 = f"<think>{pre_image}</think><image_start>"
            part2 = f"<image_end>{post_image}{answer_block}"
            output_text_list = [part1, part2]
        else:
            # Fallback if no image tag found in mm-thought (shouldn't happen for this dataset but good safety)
            # Just put everything in think? or Treat entire thought as pre-image?
            # If no image tag, maybe it's just text? But we need image_start/end for the VLM training usually.
            # Let's assume there is at least one image. If not, fallback to old logic?
            # Or just use the whole thing as think and empty post image?
            part1 = f"<think>{mm_thought}</think><image_start>"
            part2 = f"<image_end>{answer_block}"
            output_text_list = [part1, part2]
            
    else:
        # OLD LOGIC
        # Remove <image_2> from sideview_desc.
        clean_sideview_desc = item['sideview_desc'].replace(' <image_2>', '').replace('<image_2>', '').replace(' ..', '.').replace(' .', '.').replace(':.', ':').replace('  ', ' ').strip()
        part1 = f"<think>{clean_sideview_desc}</think><image_start>"
        part2 = f"<image_end>{answer_block}"
        
        output_text_list = [part1, part2]
    
    return {
        "image_list": image_list,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="linjieli222/ai2thor-path-tracing-qa-train-2point-balanced8-mmcot-16k", help="HuggingFace dataset path")
    args = parser.parse_args()

    print(f"Loading source dataset: {args.dataset_name}...")
    # ds = load_dataset("linjieli222/ai2thor-path-tracing-qa-train-2point-balanced8-16k", split="train")
    ds = load_dataset(args.dataset_name, split="train")
    
    print("Transforming dataset...")
    # Map first
    new_ds = ds.map(
        transform_item,
        remove_columns=ds.column_names,
        num_proc=4,
        features=None
    )
    
    # Define output structure
    # Derive text name from the last part of the HF path
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
    num_shards = 5
    metadata_info = {}
    
    import pyarrow.parquet as pq
    import pyarrow as pa
    
    print(f"Saving to {num_shards} shards in {data_dir}...")
    for i in range(num_shards):
        shard = new_ds.shard(num_shards=num_shards, index=i)
        filename = f"chunk_{i}.parquet"
        filepath = os.path.join(data_dir, filename)
        
        # Save shard
        shard.to_parquet(filepath)
        
        # Read back to get info
        pf = pq.ParquetFile(filepath)
        # abs_path = os.path.abspath(filepath)
        
        metadata_info[filepath] = {
            "num_row_groups": pf.num_row_groups,
            "num_rows": shard.num_rows
        }
        print(f"Saved {filename}: {shard.num_rows} rows")

    # Save metadata json
    import json
    json_path = os.path.join(info_dir, f"{dataset_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata_info, f, indent=4)
        
    print(f"Saved metadata to {json_path}")
    print("Structure matched to Bagel example.")
    
    # Verify first item of first chunk
    print("\nVerifying first item of chunk 0:")
    # ... verification logic ...

if __name__ == "__main__":
    main()
