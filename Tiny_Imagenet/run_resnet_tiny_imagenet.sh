#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
EVAL_FGSM=true
EVAL_PGD=true
AT_MODE_PGD="--at-attack pgd --at-epsilon 0.03 --at-alpha 0.01 --at-iter 7" # AT PGD specific params

# --- Base command arguments (common to all ResNet-18 runs on Tiny ImageNet) ---
BASE_ARGS="--epochs 100 --lr 0.01 --batch-size 128 --seed 42"
DATASET_ARGS="--data-dir ./data_tiny_imagenet" # Specify data directory for Tiny ImageNet
CHECKPOINT_BASE_DIR="./checkpoints_tiny_imagenet"

# --- Constructing Evaluation Flags ---
ADVERSARIAL_EVAL_FLAGS=""
if [ "$EVAL_FGSM" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-fgsm"
fi
if [ "$EVAL_PGD" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-pgd"
fi

# --- Adversarial Training Flags ---
ADVERSARIAL_TRAINING_FLAGS_PGD="--adversarial-training $AT_MODE_PGD"

# ==============================================================================
# ResNet-18 Bernoulli RAMA Experiments for Tiny ImageNet
# ==============================================================================
PYTHON_SCRIPT_BERNOULLI="Tiny_Imagenet/Resnet_multi_vector_rama_bernoulli.py"
RESNET_RAMA_BERNOULLI_ARGS="--use-rama --use-normalization --p-value 0.8 --bernoulli-values=0_1 --activation silu --sqrt-dim False"

echo "-------------------------------------------------"
echo "Starting ResNet-18 (Bernoulli RAMA) Experiments for Tiny ImageNet"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. ResNet-18 Baseline (Bernoulli script, no RAMA)
echo -e "\nRunning: 1. ResNet-18 Baseline (Bernoulli Script)"
EXP1_DIR="$CHECKPOINT_BASE_DIR/ResNet18_baseline"
mkdir -p $EXP1_DIR
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP1_DIR $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP1_DIR $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. ResNet-18 Baseline (Bernoulli Script)"
echo "-------------------------------------------------"

# # 2. ResNet-18 + AT (PGD) (Bernoulli script, no RAMA)
# echo -e "\nRunning: 2. ResNet-18 + AT (PGD) (Bernoulli Script)"
# EXP2_DIR="$CHECKPOINT_BASE_DIR/ResNet18_AT_PGD"
# mkdir -p $EXP2_DIR
# echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP2_DIR $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
# CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP2_DIR $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
# echo "Finished: 2. ResNet-18 + AT (PGD) (Bernoulli Script)"
# echo "-------------------------------------------------"

# 3. ResNet-18 + Bernoulli RAMA
echo -e "\nRunning: 3. ResNet-18 + Bernoulli RAMA"
EXP3_DIR="$CHECKPOINT_BASE_DIR/ResNet18_Bernoulli_RAMA"
mkdir -p $EXP3_DIR
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP3_DIR $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP3_DIR $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. ResNet-18 + Bernoulli RAMA"
echo "-------------------------------------------------"

# 4. ResNet-18 + Bernoulli RAMA + AT (PGD)
echo -e "\nRunning: 4. ResNet-18 + Bernoulli RAMA + AT (PGD)"
EXP4_DIR="$CHECKPOINT_BASE_DIR/ResNet18_Bernoulli_RAMA_AT_PGD"
mkdir -p $EXP4_DIR
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP4_DIR $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS --checkpoint-dir $EXP4_DIR $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. ResNet-18 + Bernoulli RAMA + AT (PGD)"
echo "-------------------------------------------------"

# # ==============================================================================
# # ResNet-18 Gaussian RAMA Experiments for Tiny ImageNet
# # ==============================================================================
# PYTHON_SCRIPT_GAUSSIAN="Tiny_Imagenet/Resnet_multi_vector_rama_gaussian.py"
# RESNET_RAMA_GAUSSIAN_ARGS="--use-rama --use-normalization --lambda-value 0.2 --activation silu --sqrt-dim False"

# echo -e "\n\n-------------------------------------------------"
# echo "Starting ResNet-18 (Gaussian RAMA) Experiments for Tiny ImageNet"
# echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
# echo "-------------------------------------------------"

# # 5. ResNet-18 Baseline (Gaussian script, no RAMA) - Note: This is effectively the same as #1 but uses the Gaussian script
# echo -e "\nRunning: 5. ResNet-18 Baseline (Gaussian Script)"
# EXP5_DIR="$CHECKPOINT_BASE_DIR/ResNet18_Gaussian_baseline"
# mkdir -p $EXP5_DIR
# echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP5_DIR $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
# CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP5_DIR $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
# echo "Finished: 5. ResNet-18 Baseline (Gaussian Script)"
# echo "-------------------------------------------------"

# # 6. ResNet-18 + AT (PGD) (Gaussian script, no RAMA)
# echo -e "\nRunning: 6. ResNet-18 + AT (PGD) (Gaussian Script)"
# EXP6_DIR="$CHECKPOINT_BASE_DIR/ResNet18_Gaussian_AT_PGD"
# mkdir -p $EXP6_DIR
# echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP6_DIR $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
# CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP6_DIR $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
# echo "Finished: 6. ResNet-18 + AT (PGD) (Gaussian Script)"
# echo "-------------------------------------------------"

# # 7. ResNet-18 + Gaussian RAMA
# echo -e "\nRunning: 7. ResNet-18 + Gaussian RAMA"
# EXP7_DIR="$CHECKPOINT_BASE_DIR/ResNet18_Gaussian_RAMA"
# mkdir -p $EXP7_DIR
# echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP7_DIR $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS"
# CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP7_DIR $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS
# echo "Finished: 7. ResNet-18 + Gaussian RAMA"
# echo "-------------------------------------------------"

# # 8. ResNet-18 + Gaussian RAMA + AT (PGD)
# echo -e "\nRunning: 8. ResNet-18 + Gaussian RAMA + AT (PGD)"
# EXP8_DIR="$CHECKPOINT_BASE_DIR/ResNet18_Gaussian_RAMA_AT_PGD"
# mkdir -p $EXP8_DIR
# echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP8_DIR $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
# CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS --checkpoint-dir $EXP8_DIR $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
# echo "Finished: 8. ResNet-18 + Gaussian RAMA + AT (PGD)"
# echo "-------------------------------------------------"

echo -e "\nAll ResNet-18 Tiny ImageNet experiments finished."
