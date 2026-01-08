import os
import sys

# Add project root to path so we can import data_processing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data_processing.verify_visualizer import verify_data

def verify_perspective_output():
    data_dir = os.path.join(project_root, "bagel_example", "editing", "perspective_qa")
    search_path = os.path.join(data_dir, "*.parquet")
    output_filename = os.path.join(project_root, "perspective_verification.html")
    
    print(f"Verifying perspective data from {search_path}...")
    verify_data(search_path, output_filename, target_total=20) # 20 samples for verification

if __name__ == "__main__":
    verify_perspective_output()
