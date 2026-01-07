#!/bin/bash
# ==============================================================================
# Prepare Checkpoint for Evaluation
# ==============================================================================
# This script automatically prepares a checkpoint for evaluation by:
# 1. Detecting checkpoint type (Path Tracing or Perspective Taking)
# 2. Copying necessary config files from a reference checkpoint if missing
# 3. Validating the checkpoint structure
#
# Usage:
#   bash scripts/prepare_checkpoint_for_eval.sh ckpt_pat/0007000
#   bash scripts/prepare_checkpoint_for_eval.sh ckpt_pet/0009000
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: Checkpoint path not provided${NC}"
    echo "Usage: bash scripts/prepare_checkpoint_for_eval.sh <checkpoint_path>"
    echo "Example: bash scripts/prepare_checkpoint_for_eval.sh ckpt_pat/0007000"
    exit 1
fi

CHECKPOINT_PATH=$1

# Get the absolute path to the repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Preparing Checkpoint for Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Checkpoint: ${GREEN}${CHECKPOINT_PATH}${NC}"
echo ""

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}❌ Error: Checkpoint directory does not exist: ${CHECKPOINT_PATH}${NC}"
    exit 1
fi

# Detect checkpoint type and set reference checkpoint
if [[ $CHECKPOINT_PATH == *"ckpt_pat"* ]]; then
    CHECKPOINT_TYPE="Path Tracing"
    REFERENCE_CKPT="ckpt_pat/0006840"
    echo -e "Type: ${GREEN}${CHECKPOINT_TYPE}${NC}"
    echo -e "Reference: ${GREEN}${REFERENCE_CKPT}${NC}"
elif [[ $CHECKPOINT_PATH == *"ckpt_pet"* ]]; then
    CHECKPOINT_TYPE="Perspective Taking"
    REFERENCE_CKPT="ckpt_pet/0003040"
    echo -e "Type: ${GREEN}${CHECKPOINT_TYPE}${NC}"
    echo -e "Reference: ${GREEN}${REFERENCE_CKPT}${NC}"
else
    echo -e "${RED}❌ Error: Cannot detect checkpoint type from path${NC}"
    echo "Path should contain 'ckpt_pat' or 'ckpt_pet'"
    exit 1
fi

# Check if reference checkpoint exists
if [ ! -d "$REFERENCE_CKPT" ]; then
    echo -e "${RED}❌ Error: Reference checkpoint does not exist: ${REFERENCE_CKPT}${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Step 1: Checking config files...${NC}"

# List of config files to check and copy
CONFIG_FILES=(
    "config.json"
    "generation_config.json"
    "llm_config.json"
    "vit_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "ae.safetensors"
)

COPIED_COUNT=0

for file in "${CONFIG_FILES[@]}"; do
    if [ ! -f "$CHECKPOINT_PATH/$file" ]; then
        if [ -f "$REFERENCE_CKPT/$file" ]; then
            echo -e "${YELLOW}⚠️  ${file} not found, copying from reference...${NC}"
            cp "$REFERENCE_CKPT/$file" "$CHECKPOINT_PATH/$file"
            COPIED_COUNT=$((COPIED_COUNT + 1))
        else
            echo -e "${YELLOW}⚠️  ${file} not found in reference checkpoint (skipping)${NC}"
        fi
    else
        echo -e "${GREEN}✅ ${file} exists${NC}"
    fi
done

if [ $COPIED_COUNT -gt 0 ]; then
    echo -e "${GREEN}📋 Copied ${COPIED_COUNT} config file(s) from reference checkpoint${NC}"
fi

echo ""
echo -e "${BLUE}Step 2: Validating checkpoint structure...${NC}"

# Check for essential model files
ESSENTIAL_FILES=(
    "config.json"
    "generation_config.json"
)

VALIDATION_PASSED=true

for file in "${ESSENTIAL_FILES[@]}"; do
    if [ ! -f "$CHECKPOINT_PATH/$file" ]; then
        echo -e "${RED}❌ Missing essential file: ${file}${NC}"
        VALIDATION_PASSED=false
    fi
done

# Check for model weights (either model.safetensors or sharded weights)
if [ -f "$CHECKPOINT_PATH/model.safetensors" ]; then
    echo -e "${GREEN}✅ Model weights found: model.safetensors${NC}"
elif [ -f "$CHECKPOINT_PATH/model.safetensors.index.json" ]; then
    echo -e "${GREEN}✅ Model weights found: sharded safetensors${NC}"
else
    echo -e "${RED}❌ Missing model weights (model.safetensors or sharded files)${NC}"
    VALIDATION_PASSED=false
fi

echo ""

if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Checkpoint is ready for evaluation!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "1. Update your Beaker YAML with: ${BLUE}CHECKPOINT_NAME=$(basename $CHECKPOINT_PATH)${NC}"
    echo -e "2. Submit evaluation: ${BLUE}beaker experiment create bagel_beaker/evaluation/eval_*.yaml${NC}"
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Checkpoint validation failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo "Please check the errors above and fix them before evaluation."
    exit 1
fi

