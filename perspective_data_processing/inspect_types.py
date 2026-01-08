from datasets import load_dataset

def inspect_types():
    print("Loading dataset: weikaih/ai2thor-perspective-qa-20k-balanced-splits")
    ds = load_dataset("weikaih/ai2thor-perspective-qa-20k-balanced-splits", split="distance_change_closer", streaming=True)
    
    types = set()
    count = 0
    max_items = 1000
    
    print(f"Inspecting first {max_items} items...")
    for item in ds:
        types.add(item['question_type'])
        count += 1
        if count >= max_items:
            break
            
    print("\nUnique Question Types:")
    for t in types:
        print(f"- {t}")

if __name__ == "__main__":
    inspect_types()
