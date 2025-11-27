#!/bin/bash
#SBATCH --account=sai.zhang
#SBATCH --qos=sai.zhang
#SBATCH --partition=hpg-turin
#SBATCH --time=1-0:00:00
#SBATCH --job-name="PRS-Net2"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1           # Request 2 GPUs
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

echo "----------------START----------------"

source /blue/sai.zhang/lihan/miniconda/bin/activate prsnet2
# source /blue/sai.zhang/lihan/miniconda/bin/activate PRS-Net
DATA_PATH="../example_data/"
DATASET=AD
SEED=22
bs=512
lr=0.0001
snp_l1=1000
init_drop=0.9
min_drop=0.0
n_layers=1
n_snp_filters=16
readout='sigmoid'


CUDA_LAUNCH_BLOCKING=1 python train.py \
    --data_path "$DATA_PATH" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --batch_size "$bs" \
    --lr "$lr" \
    --sg_l1 "$snp_l1" \
    --sg_dropout_init "$init_drop" \
    --sg_dropout_min "$min_drop" \
    --n_gnn_layers "$n_layers" \
    --n_snp_kernels "$n_snp_filters" 
    
echo "----------------END----------------"
