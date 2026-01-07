# Quick Start: Evaluate New Checkpoint

## 🚀 3-Step Evaluation Process

### For Path Tracing Checkpoint

```bash
# Step 1: Copy template
cp bagel_beaker/evaluation/eval_path_tracing_template.yaml \
   bagel_beaker/evaluation/eval_path_tracing_XXXXXXX.yaml

# Step 2: Edit YAML - change this ONE line:
#   export CHECKPOINT_NAME=XXXXXXX

# Step 3: Submit to Beaker
beaker experiment create bagel_beaker/evaluation/eval_path_tracing_XXXXXXX.yaml
```

### For Perspective Taking Checkpoint

```bash
# Step 1: Copy template
cp bagel_beaker/evaluation/eval_perspective_taking_template.yaml \
   bagel_beaker/evaluation/eval_perspective_taking_YYYYYYY.yaml

# Step 2: Edit YAML - change this ONE line:
#   export CHECKPOINT_NAME=YYYYYYY

# Step 3: Submit to Beaker
beaker experiment create bagel_beaker/evaluation/eval_perspective_taking_YYYYYYY.yaml
```

---

## 🔧 Optional: Test Locally First

```bash
# Prepare checkpoint (copies config files if needed)
bash scripts/prepare_checkpoint_for_eval.sh ckpt_pat/XXXXXXX

# Quick test with 10 samples
cd VLMEvalKit_Thinkmorph
source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/etc/profile.d/conda.sh
conda activate thinkmorph_eval
python run.py --data AI2ThorPathTracing_10 --model thinkmorph_pat_XXXXXXX --mode all
```

---

## 📊 What Gets Evaluated

| Dataset | Samples | Evaluation Time (est.) |
|---------|---------|------------------------|
| Path Tracing | 409 | ~2-3 hours |
| Perspective Taking | 800 | ~4-5 hours |

---

## 📁 Where to Find Results

**Accuracy Results:**
```
VLMEvalKit_Thinkmorph/outputs_eval/thinkmorph_pat_XXXXXXX/<timestamp>/
└── thinkmorph_pat_XXXXXXX_AI2ThorPathTracing_acc.csv
```

**Generated Visualizations:**
```
viz_outputs/pat_XXXXXXX/
```

---

## ⚠️ Important Notes

1. **Checkpoint must exist** in `ckpt_pat/` or `ckpt_pet/` directory
2. **Config files are auto-copied** - the script handles this automatically
3. **Model config must exist** in `VLMEvalKit_Thinkmorph/vlmeval/config.py`
4. **Only change CHECKPOINT_NAME** in the YAML - everything else is automatic

---

## 📝 Example: Evaluate checkpoint 0007500

```bash
# 1. Copy template
cp bagel_beaker/evaluation/eval_path_tracing_template.yaml \
   bagel_beaker/evaluation/eval_path_tracing_0007500.yaml

# 2. Edit the file - change line 18:
#    export CHECKPOINT_NAME=0007500

# 3. Submit
beaker experiment create bagel_beaker/evaluation/eval_path_tracing_0007500.yaml

# 4. Monitor
beaker experiment follow <experiment-id>
```

---

## 🆘 Troubleshooting

**"Checkpoint not found"**
→ Check path: `ls ckpt_pat/XXXXXXX/`

**"Model not found in config"**
→ Add to `VLMEvalKit_Thinkmorph/vlmeval/config.py`:
```python
"thinkmorph_pat_XXXXXXX": partial(
    ThinkMorph,
    model_path="/weka/.../ckpt_pat/XXXXXXX",
    ...
)
```

**"Missing config files"**
→ Run: `bash scripts/prepare_checkpoint_for_eval.sh ckpt_pat/XXXXXXX`

---

For detailed documentation, see [README.md](README.md)

