#!/bin/bash
# Evaluate PET No-Thought Baseline on FULL dataset (800 samples)
# Checkpoint: ckpt/pet_no_thought/run_8gpu/0004000_full
# Dataset: AI2ThorPerspective_NoArrow (full 800 samples)

set -e

cd /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/VLMEvalKit_Thinkmorph

source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/bin/activate
conda activate thinkmorph_eval

export WANDB_API_KEY=f773908953fc7bea7008ae1cf3701284de1a0682

export THINKMORPH_MODEL_PATH="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/ckpt/pet_no_thought/run_8gpu/0004000_full"
export THINKMORPH_SAVE_DIR="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/results/pet_no_thought/run_8gpu/0004000_full_eval"

echo "=========================================="
echo "PET No-Thought Baseline - FULL Evaluation"
echo "=========================================="
echo "Model: ${THINKMORPH_MODEL_PATH}"
echo "Save Dir: ${THINKMORPH_SAVE_DIR}"
echo "Dataset: AI2ThorPerspective_NoArrow (800 samples)"
echo "Mode: Text-only (no image generation)"
echo "=========================================="

python run.py \
    --data AI2ThorPerspective_NoArrow \
    --model thinkmorph_pet_no_thought \
    --mode all \
    --work-dir outputs_eval_no_thought_full \
    --verbose

echo "Evaluation complete! Results saved to: ${THINKMORPH_SAVE_DIR}"
