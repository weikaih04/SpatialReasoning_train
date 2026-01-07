# ThinkMorph Checkpoint Evaluation Guide

This directory contains templates and tools for evaluating ThinkMorph checkpoints on AI2Thor spatial reasoning datasets.

## 📋 Overview

After training produces new checkpoints, use this workflow to evaluate them on:
- **Path Tracing Dataset**: 409 samples testing spatial path understanding
- **Perspective Taking Dataset**: 800 samples testing viewpoint reasoning

## 🚀 Quick Start

### Evaluate a New Checkpoint

```bash
# 1. Copy the appropriate template
cp bagel_beaker/evaluation/eval_path_tracing_template.yaml \
   bagel_beaker/evaluation/eval_path_tracing_0007000.yaml

# 2. Edit the YAML file - change only this line:
#    export CHECKPOINT_NAME=0007000

# 3. (Optional) Test checkpoint preparation locally
bash scripts/prepare_checkpoint_for_eval.sh ckpt_pat/0007000

# 4. Submit to Beaker
beaker experiment create bagel_beaker/evaluation/eval_path_tracing_0007000.yaml
```

## 📁 Files in This Directory

### Templates
- **`eval_path_tracing_template.yaml`** - Template for Path Tracing evaluation
- **`eval_perspective_taking_template.yaml`** - Template for Perspective Taking evaluation

### Generated Files (examples)
- `eval_path_tracing_0006840.yaml` - Specific evaluation configs
- `eval_perspective_taking_0003040.yaml` - Specific evaluation configs

## 🔧 Detailed Workflow

### Step 1: Prepare Checkpoint

The preparation script automatically:
- ✅ Detects checkpoint type (Path Tracing or Perspective Taking)
- ✅ Copies missing config files from reference checkpoint
- ✅ Validates checkpoint structure

**Usage:**
```bash
# For Path Tracing checkpoints
bash scripts/prepare_checkpoint_for_eval.sh ckpt_pat/0007000

# For Perspective Taking checkpoints
bash scripts/prepare_checkpoint_for_eval.sh ckpt_pet/0009000
```

**What it copies:**
- `config.json`
- `generation_config.json`
- `llm_config.json`
- `vit_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `vocab.json`, `merges.txt`
- `ae.safetensors`

### Step 2: Create Evaluation YAML

**For Path Tracing:**
```bash
cp bagel_beaker/evaluation/eval_path_tracing_template.yaml \
   bagel_beaker/evaluation/eval_path_tracing_XXXXXXX.yaml
```

**For Perspective Taking:**
```bash
cp bagel_beaker/evaluation/eval_perspective_taking_template.yaml \
   bagel_beaker/evaluation/eval_perspective_taking_YYYYYYY.yaml
```

### Step 3: Modify Checkpoint Name

Edit the copied YAML file and change **only this line**:

```yaml
export CHECKPOINT_NAME=0007000  # 👈 Change this to your checkpoint number
```

Everything else is automatically configured!

### Step 4: Submit to Beaker

```bash
beaker experiment create bagel_beaker/evaluation/eval_path_tracing_0007000.yaml
```

## 📊 Datasets

| Dataset Name | Samples | Description |
|-------------|---------|-------------|
| `AI2ThorPathTracing` | 409 | Full Path Tracing evaluation |
| `AI2ThorPathTracing_10` | 10 | Quick test (for debugging) |
| `AI2ThorPerspective_NoArrow` | 800 | Full Perspective Taking evaluation |
| `AI2ThorPerspective_NoArrow_10` | 10 | Quick test (for debugging) |

## 📂 Output Locations

After evaluation completes:

**Results:**
```
VLMEvalKit_Thinkmorph/outputs_eval/
├── thinkmorph_pat_0007000/
│   └── <timestamp>/
│       └── thinkmorph_pat_0007000_AI2ThorPathTracing_acc.csv
└── thinkmorph_pet_0009000/
    └── <timestamp>/
        └── thinkmorph_pet_0009000_AI2ThorPerspective_NoArrow_acc.csv
```

**Visualizations:**
```
viz_outputs/
├── pat_0007000/  # Generated images for Path Tracing
└── pet_0009000/  # Generated images for Perspective Taking
```

## ⚙️ Configuration Parameters

The templates use these default settings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gpuCount` | 2 | Number of GPUs |
| `sharedMemory` | 100GiB | Shared memory allocation |
| `priority` | normal | Job priority |
| `preemptible` | true | Can be preempted |
| `temperature` | 0.3 | Sampling temperature |
| `max_think_token_n` | 4096 | Max thinking tokens |
| `num_timesteps` | 50 | Diffusion steps |
| `image_resolution` | 1024 | Image resolution |

To modify these, edit the model configuration in `VLMEvalKit_Thinkmorph/vlmeval/config.py`.

## 🔍 Troubleshooting

### Checkpoint preparation fails
```bash
# Check if checkpoint directory exists
ls -la ckpt_pat/0007000/

# Check if reference checkpoint has config files
ls -la ckpt_pat/0006840/
```

### Evaluation fails to start
- Verify conda environment exists: `conda env list | grep thinkmorph_eval`
- Check GPU availability in Beaker cluster
- Review Beaker logs for detailed error messages

### Model not found in config.py
The YAML templates expect model names like `thinkmorph_pat_XXXXXX` to be defined in `VLMEvalKit_Thinkmorph/vlmeval/config.py`. 

Add new models to the `thinkmorph_series` dictionary:
```python
"thinkmorph_pat_0007000": partial(
    ThinkMorph,
    model_path="/weka/.../ckpt_pat/0007000",
    think=True,
    understanding_output=False,
    temperature=0.3,
    max_think_token_n=4096,
    save_dir="/weka/.../viz_outputs/pat_0007000"
),
```

## 📝 Example: Complete Evaluation Flow

```bash
# 1. New checkpoint created during training
# ckpt_pat/0007500/

# 2. Prepare checkpoint
bash scripts/prepare_checkpoint_for_eval.sh ckpt_pat/0007500

# 3. Create evaluation YAML
cp bagel_beaker/evaluation/eval_path_tracing_template.yaml \
   bagel_beaker/evaluation/eval_path_tracing_0007500.yaml

# 4. Edit YAML: change CHECKPOINT_NAME=0007500

# 5. Submit evaluation
beaker experiment create bagel_beaker/evaluation/eval_path_tracing_0007500.yaml

# 6. Monitor progress
beaker experiment follow <experiment-id>

# 7. View results
cat VLMEvalKit_Thinkmorph/outputs_eval/thinkmorph_pat_0007500/*/thinkmorph_pat_0007500_AI2ThorPathTracing_acc.csv
```

## 🎯 Tips

- **Quick testing**: Use `_10` datasets for fast validation before full evaluation
- **Batch evaluation**: Create multiple YAML files and submit them in parallel
- **Resource optimization**: Adjust `gpuCount` based on availability
- **Result tracking**: Keep YAML files for reproducibility and tracking

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Beaker experiment logs
3. Verify checkpoint structure with `prepare_checkpoint_for_eval.sh`

