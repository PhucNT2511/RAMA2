# CUDA_VISIBLE_DEVICES=0 python Cifar10/SwinT_multi_vector_rama_bernoulli.py --epochs 100 --batch-size 128 --use-rama --use-normalization --p-value 0.6 --bernoulli-values="-1_1" --activation leaky_relu
# CUDA_VISIBLE_DEVICES=0 python Cifar10/SwinT_multi_vector_rama_bernoulli.py --epochs 100 --batch-size 128 --use-rama --use-normalization --use-hyperparameter-optimization --p-value 0.6 --bernoulli-values="-1_1" --activation leaky_relu
# CUDA_VISIBLE_DEVICES=0 python Cifar10/SwinT_multi_vector_rama_bernoulli.py --epochs 200 --use-rama --use-normalization --p-value 0.4 --bernoulli-values="-1_1" --activation leaky_relu --optimize-every 10

CUDA_VISIBLE_DEVICES=0 python Cifar10/EffficientNet_multi_vector_rama_bernoulli.py --epochs 100 --use-rama --use-normalization --p-value 0.5 --bernoulli-values="0_1" --activation leaky_relu --use-hyperparameter-optimization --optimize-every 10
CUDA_VISIBLE_DEVICES=0 python Cifar10/Resnet_multi_vector_rama_bernoulli.py --epochs 100 --use-rama --use-normalization --p-value 0.5 --bernoulli-values="0_1" --activation leaky_relu --use-hyperparameter-optimization --optimize-every 10
CUDA_VISIBLE_DEVICES=0 python Cifar10/SwinT_multi_vector_rama_bernoulli.py --epochs 100 --use-rama --use-normalization --p-value 0.5 --bernoulli-values="0_1" --activation leaky_relu --use-hyperparameter-optimization --optimize-every 10
