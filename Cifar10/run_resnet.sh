#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
# These will be used for all ResNet-18 runs to fill the table
EVAL_FGSM=true
EVAL_PGD=true

# --- Base command arguments (common to all ResNet-18 runs) ---
PYTHON_SCRIPT="Cifar10/Resnet_multi_vector_rama_bernoulli.py"
# Adjust base arguments like epochs, lr, batch_size as needed for ResNet-18.
# The ResNet script has default lr=0.1, epochs=100. Let's use that.
BASE_ARGS="--epochs 100 --lr 0.1 --lr-scheduler cosine" 

# --- RAMA specific arguments for ResNet-18 ---
# Adjust these as needed for your desired ResNet-18 + RAMA configuration.
# These are example values, check Resnet_multi_vector_rama_bernoulli.py for defaults and options.
RESNET_RAMA_ARGS="--use-rama --use-normalization --p-value 0.7 --bernoulli-values=-1_1 --activation silu --rama-positions final"

# --- Attack parameters for AT and Evaluation (defaults from the Python script will be used if not specified here) ---
# ADV_EPSILON="8/255"
# PGD_ALPHA="2/255"
# PGD_ITER="10" # For PGD AT, 7-10 iterations. For PGD evaluation, 10-20.
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

# --- Experiment Section: ResNet-18 --- 
echo "-------------------------------------------------"
echo "Starting ResNet-18 Experiments for CIFAR-10 Table"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. ResNet-18 Baseline
echo "
Running: 1. ResNet-18 Baseline"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. ResNet-18 Baseline"
echo "-------------------------------------------------"

# 2. ResNet-18 + AT (Using PGD for Adversarial Training)
AT_MODE_PGD="pgd"
ADVERSARIAL_TRAINING_FLAGS_PGD="--adversarial-training $AT_MODE_PGD"
echo "
Running: 2. ResNet-18 + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. ResNet-18 + AT (PGD)"
echo "-------------------------------------------------"

# 3. ResNet-18 + RAMA
echo "
Running: 3. ResNet-18 + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. ResNet-18 + RAMA"
echo "-------------------------------------------------"

# 4. ResNet-18 + AT (PGD) + RAMA
echo "
Running: 4. ResNet-18 + AT (PGD) + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. ResNet-18 + AT (PGD) + RAMA"
echo "-------------------------------------------------"

echo "
All ResNet-18 experiments finished."