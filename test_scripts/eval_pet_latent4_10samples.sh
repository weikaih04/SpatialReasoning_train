#!/bin/bash
# Evaluation script for PET Latent4 (64x64 output) - 10 samples

cd /weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/VLMEvalKit_Thinkmorph

source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/bin/activate
conda activate thinkmorph_eval

export WANDB_API_KEY=f773908953fc7bea7008ae1cf3701284de1a0682
export THINKMORPH_MODEL_PATH="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/ckpt/pet_latent4/run_8gpu/0010000_full"
export THINKMORPH_IMAGE_RESOLUTION=64
export THINKMORPH_SAVE_DIR="/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/results/pet_latent4/run_8gpu/0010000_10samples"

echo "=========================================="
echo "Evaluating PET Latent4 (64x64)"
echo "Model: ${THINKMORPH_MODEL_PATH}"
echo "Resolution: ${THINKMORPH_IMAGE_RESOLUTION}"
echo "Save Dir: ${THINKMORPH_SAVE_DIR}"
echo "=========================================="

python run.py --data AI2ThorPerspective_NoArrow_10 --model thinkmorph_pet --mode all --work-dir outputs_eval_latent4 --verbose
