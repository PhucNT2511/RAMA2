#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
# These will be used for all Swin-T runs to fill the table
EVAL_FGSM=true
EVAL_PGD=true

# --- Base command arguments (common to all Swin-T runs) ---
# Assuming SwinT_multi_vector_rama_bernoulli.py is the script for Swin-T
PYTHON_SCRIPT="Cifar10/SwinT_multi_vector_rama_bernoulli.py"
BASE_ARGS="--epochs 100" # You might want to adjust epochs or other base params like LR, batch size

# --- RAMA specific arguments for Swin-T ---
# Using the RAMA args from your previous script for Example 2
# Adjust these as needed for your desired Swin-T + RAMA configuration
SWIN_T_RAMA_ARGS="--use-rama --use-normalization --p-value 0.7 --bernoulli-values=-1_1 --activation silu --rama-positions final"

# --- Attack parameters for AT and Evaluation (defaults from the Python script will be used if not specified here) ---
# ADV_EPSILON="8/255"
# PGD_ALPHA="2/255"
# PGD_ITER="10" # For PGD AT, 7-10 iterations are common. For PGD evaluation, 10-20 are common.
# If overriding, add them to BASE_ARGS or specific command lines:
# مثلاً: BASE_ARGS+=" --adv-epsilon $ADV_EPSILON --pgd-alpha $PGD_ALPHA --pgd-iter $PGD_ITER"

# --- Constructing Evaluation Flags (always on for these experiments) ---
ADVERSARIAL_EVAL_FLAGS=""
if [ "$EVAL_FGSM" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-fgsm"
fi
if [ "$EVAL_PGD" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-pgd"
fi

# --- Experiment Section: Swin-T --- 
echo "-------------------------------------------------"
echo "Starting Swin-T Experiments for CIFAR-10 Table"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. Swin-T Baseline
echo "
Running: 1. Swin-T Baseline"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. Swin-T Baseline"
echo "-------------------------------------------------"

# 2. Swin-T + AT (Using PGD for Adversarial Training)
AT_MODE_PGD="pgd"
ADVERSARIAL_TRAINING_FLAGS_PGD="--adversarial-training $AT_MODE_PGD"
echo "
Running: 2. Swin-T + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. Swin-T + AT (PGD)"
echo "-------------------------------------------------"

# 3. Swin-T + RAMA
echo "
Running: 3. Swin-T + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWIN_T_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWIN_T_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. Swin-T + RAMA"
echo "-------------------------------------------------"

# 4. Swin-T + AT (PGD) + RAMA
echo "
Running: 4. Swin-T + AT (PGD) + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWIN_T_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWIN_T_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. Swin-T + AT (PGD) + RAMA"
echo "-------------------------------------------------"

echo "
All Swin-T experiments finished."