#!/bin/bash
# Evaluate PET Latent16 on FULL dataset (800 samples)
# Checkpoint: ckpt/pet_latent16/run_8gpu/0010000_full
# Dataset: AI2ThorPerspective_NoArrow (full 800 samples)
# Resolution: 256x256

set -e

cd /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/VLMEvalKit_Thinkmorph

source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/bin/activate
conda activate thinkmorph_eval

export WANDB_API_KEY=f773908953fc7bea7008ae1cf3701284de1a0682

export THINKMORPH_MODEL_PATH="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/ckpt/pet_latent16/run_8gpu/0010000_full"
export THINKMORPH_IMAGE_RESOLUTION=256
export THINKMORPH_SAVE_DIR="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/results/pet_latent16/run_8gpu/0010000_full_eval"

echo "=========================================="
echo "PET Latent16 - FULL Evaluation"
echo "=========================================="
echo "Model: ${THINKMORPH_MODEL_PATH}"
echo "Resolution: ${THINKMORPH_IMAGE_RESOLUTION}x${THINKMORPH_IMAGE_RESOLUTION}"
echo "Save Dir: ${THINKMORPH_SAVE_DIR}"
echo "Dataset: AI2ThorPerspective_NoArrow (800 samples)"
echo "=========================================="

python run.py \
    --data AI2ThorPerspective_NoArrow \
    --model thinkmorph_pet \
    --mode all \
    --work-dir outputs_eval_latent16_full \
    --verbose

echo "Evaluation complete! Results saved to: ${THINKMORPH_SAVE_DIR}"
