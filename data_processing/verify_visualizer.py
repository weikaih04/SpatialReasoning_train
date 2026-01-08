import pyarrow.parquet as pq
import glob
import pandas as pd
import random
import base64
import os
import html
import argparse

def verify_data(search_path, output_filename, target_total=50):
    files = glob.glob(search_path, recursive=True)
    if not files:
        print(f"No parquet files found matching: {search_path}")
        return
        
    files.sort()
    print(f"Found {len(files)} parquet files. Sampling evenly...")

    collected_dfs = []
    # Calculate how many to take from each file
    samples_per_file = max(1, int(target_total / len(files)) + 1)

    random.seed(42)

    for fpath in files:
        try:
            pf = pq.ParquetFile(fpath)
            if pf.num_row_groups > 0:
                # Read a random row group to avoid loading the entire large file
                rg_index = random.randint(0, pf.num_row_groups - 1)
                df_chunk = pf.read_row_group(rg_index).to_pandas()

                n_take = min(samples_per_file, len(df_chunk))
                if n_take > 0:
                    collected_dfs.append(df_chunk.sample(n=n_take, random_state=42))
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not collected_dfs:
        print("No data collected.")
        return

    df_sample = pd.concat(collected_dfs, ignore_index=True)

    # Shuffle combined results
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    # Limit to target if slightly over
    if len(df_sample) > target_total:
        df_sample = df_sample.head(target_total)

    print(f"Total sampled rows: {len(df_sample)}")
    
    html_content = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
            tr:nth-child(even){background-color: #f2f2f2;}
            img { max-width: 300px; max-height: 300px; display: block; margin: 5px; border: 1px solid #ccc; }
            .text-block { white-space: pre-wrap; font-family: monospace; max-width: 500px; }
            .container { display: flex; flex-wrap: wrap; }
        </style>
    </head>
    <body>
    <h1>Dataset Verification</h1>
    <table>
    <tr>
        <th>ID</th>
        <th>Images</th>
        <th>Instruction</th>
        <th>Output Text</th>
    </tr>
    """
    
    for idx, row in df_sample.iterrows():
        # Images
        images_html = "<div class='container'>"
        image_list = row["image_list"]
        for i, img_bytes in enumerate(image_list):
            if isinstance(img_bytes, bytes):
                try:
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    images_html += f"<div><strong>Img {i}</strong><br><img src='data:image/png;base64,{b64_img}'/></div>"
                except Exception as e:
                    images_html += f"<div><strong>Img {i}</strong><br>Error encoding: {e}</div>"
            else:
                 images_html += f"<div><strong>Img {i}</strong><br>Not bytes</div>"
        images_html += "</div>"
            
        # Instruction
        instr = row["instruction_list"][0] if len(row["instruction_list"]) > 0 else "N/A"
        
        # Output
        out_list = row["output_text_list"]
        out_html = ""
        for i, txt in enumerate(out_list):
            # Escape HTML tags like <think>
            safe_txt = html.escape(txt)
            out_html += f"<div class='text-block'><strong>Part {i}:</strong><br>{safe_txt}</div>"

        html_content += f"""
        <tr>
            <td>{idx}</td>
            <td>{images_html}</td>
            <td><div class='text-block'>{html.escape(instr)}</div></td>
            <td>{out_html}</td>
        </tr>
        """
        
    html_content += "</table></body></html>"
    
    with open(output_filename, "w") as f:
        f.write(html_content)
        
    print(f"Saved HTML verification to {os.path.abspath(output_filename)}")

def main():
    parser = argparse.ArgumentParser(description="Verify dataset by generating HTML visualization.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    # Default to original path
    default_search_path = os.path.join(project_root, "bagel_example/editing/path-tracing-2point-balanced8-16k/*.parquet")
    default_output = os.path.join(project_root, "verification_output.html")

    parser.add_argument("--data_pattern", type=str, default=default_search_path, help="Glob pattern for parquet files")
    parser.add_argument("--output", type=str, default=default_output, help="Output HTML file path")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to visualize")

    args = parser.parse_args()
    
    verify_data(args.data_pattern, args.output, args.samples)

if __name__ == "__main__":
    main()
