import os
from datasets import load_dataset, Dataset

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

    instruction_list = [clean_question]
    
    # Prepare output_text_list
    # Format: 
    # 1. <think>{sideview_desc}</think><image_start>
    # 2. <image_end><answer>{answer}</answer>
    
    # Remove <image_2> from sideview_desc.
    clean_sideview_desc = item['sideview_desc'].replace(' <image_2>', '').replace('<image_2>', '').replace(' ..', '.').replace(' .', '.').replace(':.', ':').replace('  ', ' ').strip()
    answer = item['answer']
    
    part1 = f"<think>{clean_sideview_desc}</think><image_start>"
    part2 = f"<image_end><answer>{answer}</answer>"
    
    output_text_list = [part1, part2]
    
    return {
        "image_list": image_list,
        "instruction_list": instruction_list,
        "output_text_list": output_text_list
    }

def main():
    print("Loading source dataset...")
    # Load from HF
    ds = load_dataset("linjieli222/ai2thor-path-tracing-qa-train-2point-balanced8-16k", split="train")
    
    print("Selecting subset for debug...")
    ds = ds.select(range(50))
    
    print("Transforming dataset...")
    new_ds = ds.map(
        transform_item, 
        remove_columns=ds.column_names, 
        num_proc=4,
        features=None  # Let HF infer types (likely Sequence(Value("binary")) for image_list)
    )
    
    print("Saving transformed dataset...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_path = os.path.join(project_root, "transformed_bagel_dataset")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save as Parquet
    output_file = os.path.join(save_path, "train.parquet")
    # For a small subset, this will be fast
    new_ds.to_parquet(output_file)
    
    print(f"Done. Saved to {output_file}")
    print(f"Features: {new_ds.features}")

if __name__ == "__main__":
    main()
