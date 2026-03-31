#!/bin/bash
set -e

PHYSICS_MODEL="QCD"          # Options: "QED" or "QCD"
MODE="jepa"              # Options: "baseline" (lambda=0) or "jepa" (lambda=1)
NOTATION="infix"             # Options: "infix" (Standard Math) or "prefix" (Polish)
AUG_MULT=15             # 1 = Baseline (No Aug), 15 = 15x Augmentation
MAX_LEN=512                # Context window size
RAW_DATA_DIR="/media/kavinder/hdd/Arnabi/SYMBA-Test Data"
WT_PATH="./logs/${MODE}/checkpoints/${PHYSICS_MODEL}/best_model_${PHYSICS_MODEL}_${MODE}_${NOTATION}_mult${AUG_MULT}.pt"

echo "$PHYSICS_MODEL | $MODE | $NOTATION | ${AUG_MULT}x Aug"

echo -e "\n Running Data Preprocessing"
python scripts/01_preprocess_data.py \
    --raw_data_dir "$RAW_DATA_DIR" \
    --output_dir "./data/processed_${AUG_MULT}x" \
    --augment_multiplier $AUG_MULT

echo -e "\n Building Vocabulary ($NOTATION)"
python scripts/02_build_vocab.py \
    --physics_model $PHYSICS_MODEL \
    --notation $NOTATION \
    --augment_multiplier $AUG_MULT

echo -e "\n Training Model"
python scripts/03_train_jepa.py \
    --physics_model $PHYSICS_MODEL \
    --mode $MODE \
    --notation $NOTATION \
    --augment_multiplier $AUG_MULT \

echo -e "\n Evaluating Model"
python inference/evaluate.py \
    --physics_model $PHYSICS_MODEL \
    --mode $MODE \
    --notation $NOTATION \
    --augment_multiplier $AUG_MULT \
    --max_len $MAX_LEN \
    --wt_path "$WT_PATH"
