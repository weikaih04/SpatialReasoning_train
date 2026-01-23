#!/bin/bash
# Evaluate Sideview-to-PAT 17k on Path Tracing (10 samples)
# Checkpoint: ckpt/sideview_to_pat/run_8gpu/0017000_full
# Dataset: AI2ThorPathTracing_10 (10 samples)

set -e

cd /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/VLMEvalKit_Thinkmorph

source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/bin/activate
conda activate thinkmorph_eval

export WANDB_API_KEY=f773908953fc7bea7008ae1cf3701284de1a0682

export THINKMORPH_MODEL_PATH="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/ckpt/sideview_to_pat/run_8gpu/0017000_full"
export THINKMORPH_IMAGE_RESOLUTION=1024
export THINKMORPH_SAVE_DIR="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/results/sideview_to_pat/run_8gpu/0017000_10samples"

echo "=========================================="
echo "Sideview-to-PAT 17k Path Tracing Evaluation"
echo "=========================================="
echo "Model: ${THINKMORPH_MODEL_PATH}"
echo "Resolution: ${THINKMORPH_IMAGE_RESOLUTION}"
echo "Save Dir: ${THINKMORPH_SAVE_DIR}"
echo "Dataset: AI2ThorPathTracing_10 (10 samples)"
echo "=========================================="

python run.py \
    --data AI2ThorPathTracing_10 \
    --model thinkmorph_pat \
    --mode all \
    --work-dir outputs_eval_sideview_to_pat_17k \
    --verbose

echo "Evaluation complete! Results saved to: ${THINKMORPH_SAVE_DIR}"
