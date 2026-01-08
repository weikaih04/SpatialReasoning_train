from datasets import load_dataset
import pandas as pd

def inspect_new_dataset():
    print("Loading dataset: weikaih/ai2thor-perspective-qa-20k-balanced-splits")
    try:
        ds = load_dataset("weikaih/ai2thor-perspective-qa-20k-balanced-splits", split="distance_change_closer", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\nRetrieving first item...")
    item = next(iter(ds))
    
    print("\nKeys:", item.keys())
    
    print("\nSample Data (First Item):")
    for k, v in item.items():
        if k == 'image' or 'image' in k:
            print(f"{k}: <Image Object> (Size/Mode would need PIL inspection)")
        else:
            print(f"{k}: {v}")
            
    # Inspect type of images more closely if possible
    # (Assuming we might have similar fields like topdown or sideview)

if __name__ == "__main__":
    inspect_new_dataset()
