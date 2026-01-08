import sys
import os
import torch
import pyarrow.parquet as pq
import json

# Add project root to path so we can import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data.interleave_datasets.edit_dataset import UnifiedEditIterableDataset

class MockTokenizer:
    def encode(self, text):
        # Return dummy ids
        return [1, 2, 3, 4]

def mock_transform(image):
    # Return dummy tensor
    # VAE transform usually returns [C, H, W]
    return torch.randn(3, 256, 256)

def mock_vit_transform(image):
    # ViT transform usually returns [C, H, W]
    return torch.randn(3, 224, 224)

# Mock ImageTransform class if needed (dataset_base checks for stride)
class MockImageTransform:
    def __init__(self, stride=16):
        self.stride = stride
    def __call__(self, image):
        # We need to return a tensor that has a shape divisible by stride
        return torch.randn(3, 256, 256)

class MockViTImageTransform:
    def __init__(self, stride=14):
        self.stride = stride
    def __call__(self, image):
        return torch.randn(3, 224, 224)

def verify_loader():
    # project_root is already calculated above when we added it to sys.path, but we are inside function scope.
    # Re-calculate or use global if available. Better re-calculate or pass.
    # But wait, verify_loader doesn't take args.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, "transformed_bagel_dataset")
    parquet_path = os.path.join(dataset_dir, "train.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found at {parquet_path}")
        return

    # Create parquet_info
    print("Generating parquet info...")
    pf = pq.ParquetFile(parquet_path)
    num_row_groups = pf.num_row_groups
    parquet_info = {
        parquet_path: {
            "num_row_groups": num_row_groups
        }
    }
    
    print("Instantiating dataset...")
    
    # Mock transforms with stride attribute
    transform = MockImageTransform(stride=16)
    vit_transform = MockViTImageTransform(stride=14)
    
    ds = UnifiedEditIterableDataset(
        dataset_name="unified_edit",
        transform=transform,
        tokenizer=MockTokenizer(),
        vit_transform=vit_transform,
        data_dir_list=[dataset_dir],
        num_used_data=[50],
        parquet_info=parquet_info,
        local_rank=0,
        world_size=1,
        num_workers=0 # Use 0 for main process debugging
    )
    
    print("Iterating dataset...")
    try:
        data_iter = iter(ds)
        item = next(data_iter)
        print("Successfully loaded first item!")
        print("Keys:", item.keys())
        print("Sequence plan length:", len(item['sequence_plan']))
        # print first few plan items
        for i, plan in enumerate(item['sequence_plan']):
            print(f"Plan step {i}: {plan['type']}, loss={plan['loss']}")
            
    except Exception as e:
        print(f"FAILED to load item. Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_loader()
