# Data Transformation Scripts

This directory contains scripts for transforming and verifying the Bagel dataset for ThinkMorph.

## Contents

-   `transform_data.py`: Main script to transform the entire dataset.
-   `transform_debug.py`: Debug script to transform a small subset (50 items) for quick testing.
-   `verify_visualizer.py`: Verifies the text and image content of the transformed parquet files. Generates an HTML report.
-   `verify_dataset_loader.py`: Verifies that the `UnifiedEditIterableDataset` loader can load the transformed data.
-   `verify_parquet.py`: Low-level verification of parquet file structure.

## Usage

You can run these scripts from the project root or from this directory.

### 1. Debug Transformation

Run this to generate a small sample dataset in `transformed_bagel_dataset/` at the project root.

```bash
python data_processing/transform_debug.py
```

### 2. Verify Transformation

After running the transformation, verify the output.

```bash
python data_processing/verify_visualizer.py
```

This will generate `verification_output.html` in the project root.

### 3. Verify Loader

Ensure the PyTorch dataset loader works with the generated data.

```bash
python data_processing/verify_dataset_loader.py
```

### 4. Full Transformation

To run the full transformation (this takes longer):

```bash
python data_processing/transform_data.py
```
