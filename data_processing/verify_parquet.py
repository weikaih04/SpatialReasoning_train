import pyarrow.parquet as pq
import os
import glob
import io
from PIL import Image

def verify_parquet():
    # Find a parquet file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    search_path = os.path.join(project_root, "transformed_bagel_dataset/**/*.parquet")
    files = glob.glob(search_path, recursive=True)
    if not files:
        print("No parquet files found in transformed_bagel_dataset")
        return
    
    fpath = files[0]
    print(f"Inspecting {fpath}")
    
    fs = pq.ParquetFile(fpath)
    df = fs.read_row_group(0).to_pandas()
    row = df.iloc[0]
    
    print("Keys:", row.keys())
    images = row["image_list"]
    print("image_list type:", type(images))
    if isinstance(images, (list, tuple,  np.ndarray if 'np' in globals() else object)):
        print("image_list length:", len(images))
        if len(images) > 0:
            elem = images[0]
            print("Element type:", type(elem))
            print("Element value summary:", str(elem)[:100])
            
            # Check if it works with the training code logic
            try:
                # Training code expectation:
                # pil_img2rgb(Image.open(io.BytesIO(images[0])))
                img = Image.open(io.BytesIO(elem))
                print("SUCCESS: Element is valid bytes for Image.open")
            except Exception as e:
                print(f"FAILURE: Element is NOT valid bytes for Image.open. Error: {e}")
                
            if isinstance(elem, dict):
                 print("Element is a dict. Keys:", elem.keys())

if __name__ == "__main__":
    verify_parquet()
