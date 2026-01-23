"""
Verify the transformed multi-view counting parquet files.

Usage:
    python multi_view_counting_data_processing/verify_multi_view_counting.py
"""

import os
import pyarrow.parquet as pq


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(
        project_root,
        "bagel_example",
        "editing",
        "multi_view_counting_training_v5"
    )

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Run transform_multi_view_counting.py first.")
        return

    print(f"Verifying data in: {data_dir}")
    print("=" * 60)

    # Find all parquet files
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])

    if not parquet_files:
        print("No parquet files found!")
        return

    total_rows = 0
    for filename in parquet_files:
        filepath = os.path.join(data_dir, filename)
        pf = pq.ParquetFile(filepath)
        num_rows = pf.metadata.num_rows
        total_rows += num_rows
        print(f"{filename}: {num_rows} rows")

    print(f"\nTotal: {total_rows} rows across {len(parquet_files)} files")
    print("=" * 60)

    # Verify first item
    print("\nVerifying first item from chunk_0.parquet:")
    first_chunk = os.path.join(data_dir, "chunk_0.parquet")
    pf = pq.ParquetFile(first_chunk)

    # Read columns
    batch = pf.read_row_group(0, columns=[
        'instruction_list',
        'output_text_list',
        'num_input_images'
    ])

    instrs = batch.column('instruction_list').to_pylist()
    outputs = batch.column('output_text_list').to_pylist()
    num_inputs = batch.column('num_input_images').to_pylist()

    print(f"\n[Row 0]")
    print(f"instruction starts with system prompt: {instrs[0][0][:50]}...")
    print(f"output_text_list: {outputs[0]}")
    print(f"num_input_images: {num_inputs[0]}")

    # Verify schema
    print("\n[Schema]")
    print(pf.schema_arrow)

    # Check key properties
    print("\n[Validation]")

    # Check system prompt
    if "Let's think step by step" in instrs[0][0]:
        print("✓ System prompt present")
    else:
        print("✗ System prompt MISSING")

    # Check num_input_images
    if num_inputs[0] == 4:
        print("✓ num_input_images = 4")
    else:
        print(f"✗ num_input_images = {num_inputs[0]} (expected 4)")

    # Check output text format
    if "<think>" in outputs[0][0] and "<image_start>" in outputs[0][0]:
        print("✓ output_text[0] has <think> and <image_start>")
    else:
        print("✗ output_text[0] format incorrect")

    if "<image_end>" in outputs[0][1] and "<answer>" in outputs[0][1]:
        print("✓ output_text[1] has <image_end> and <answer>")
    else:
        print("✗ output_text[1] format incorrect")


if __name__ == "__main__":
    main()
