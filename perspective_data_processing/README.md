# Perspective QA Processing Scripts

This directory contains scripts for transforming and verifying the Perspective QA dataset for ThinkMorph.

## Contents

-   `transform_perspective.py`: Main script to transform the dataset. Supports both full and debug modes.
-   `verify_perspective.py`: Verifies the content of the transformed parquet files.
-   `debug_regex.py`: Standalone script to test regex logic for template extraction.
-   `inspect_ds.py`: Helper to inspect the raw source dataset structure.
-   `inspect_types.py`: Helper to analyze question types in the source dataset.

## Usage

### 1. Debug Transformation

Run on a small subset (10 items per split) to verify logic.

```bash
python perspective_data_processing/transform_perspective.py --debug
```

This will create `perspective_bagel_dataset/` in the project root.

### 2. Verify Output

Check the transformed parquet files and generate an HTML visualization report (`perspective_verification.html`).

```bash
python perspective_data_processing/verify_perspective.py
```

### 3. Full Transformation

Run on the full dataset.

```bash
python perspective_data_processing/transform_perspective.py
```
