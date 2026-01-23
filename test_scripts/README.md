# Test Scripts

This directory contains all testing scripts organized by component.

## Directory Structure

- `test_dataloader/` - Scripts for testing data loading and preprocessing
- `test_model/` - Scripts for testing model inference and generation
- `test_evaluation/` - Scripts for testing evaluation pipelines

## Usage

All test scripts should:
1. Be self-contained and runnable with `python test_script.py`
2. Use argparse for configuration
3. Save outputs to timestamped directories
4. Log execution details

## Environment

Use `conda activate thinkmorph` for all test scripts.
