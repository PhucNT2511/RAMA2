#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
# These will be used for all EfficientNet-B2 runs to fill the table
EVAL_FGSM=true
EVAL_PGD=true

# --- Base command arguments (common to all EfficientNet-B2 runs) ---
PYTHON_SCRIPT="Cifar10/EfficientNet_multi_vector_rama_bernoulli.py" # Corrected spelling
# Defaults from EffficientNet script: epochs 20, lr 0.01. Optimizer is SGD.
BASE_ARGS="--epochs 100 --lr 0.01" 

# --- RAMA specific arguments for EfficientNet-B2 ---
# Adjust these as needed for your desired EfficientNet-B2 + RAMA configuration.
# Check EfficientNet_multi_vector_rama_bernoulli.py for defaults and options.
# Assuming Bernoulli RAMA as per current script capabilities.
EFFICIENTNET_RAMA_ARGS="--use-rama --rama-type bernoulli --use-normalization --p-value 0.5 --lambda-value 1.0 --bernoulli-values=\"-1_1\" --activation silu"
# Note: EfficientNet script does not seem to have explicit --rama-positions argument like ResNet/SwinT.
# RAMA application in EfficientNet might be hardcoded or follow a different logic based on its structure.

# --- Attack parameters for AT and Evaluation (defaults from the Python script will be used if not specified here) ---
# ADV_EPSILON="8/255"
# PGD_ALPHA="2/255"
# PGD_ITER="10" 
# If overriding, add them to BASE_ARGS or specific command lines:
# e.g., BASE_ARGS+=" --adv-epsilon $ADV_EPSILON --pgd-alpha $PGD_ALPHA --pgd-iter $PGD_ITER"

# --- Constructing Evaluation Flags (always on for these experiments) ---
ADVERSARIAL_EVAL_FLAGS=""
if [ "$EVAL_FGSM" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-fgsm"
fi
if [ "$EVAL_PGD" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-pgd"
fi

# --- Experiment Section: EfficientNet-B2 --- 
echo "-------------------------------------------------"
echo "Starting EfficientNet-B2 Experiments for CIFAR-10 Table"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. EfficientNet-B2 Baseline
echo "
Running: 1. EfficientNet-B2 Baseline"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. EfficientNet-B2 Baseline"
echo "-------------------------------------------------"

# 2. EfficientNet-B2 + AT (Using PGD for Adversarial Training)
AT_MODE_PGD="pgd"
ADVERSARIAL_TRAINING_FLAGS_PGD="--adversarial-training $AT_MODE_PGD"
echo "
Running: 2. EfficientNet-B2 + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. EfficientNet-B2 + AT (PGD)"
echo "-------------------------------------------------"

# 3. EfficientNet-B2 + RAMA
echo "
Running: 3. EfficientNet-B2 + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $EFFICIENTNET_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $EFFICIENTNET_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. EfficientNet-B2 + RAMA"
echo "-------------------------------------------------"

# 4. EfficientNet-B2 + AT (PGD) + RAMA
echo "
Running: 4. EfficientNet-B2 + AT (PGD) + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $EFFICIENTNET_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $EFFICIENTNET_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. EfficientNet-B2 + AT (PGD) + RAMA"
echo "-------------------------------------------------"

echo "
All EfficientNet-B2 experiments finished."