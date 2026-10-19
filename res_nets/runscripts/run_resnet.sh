#!/bin/bash

#SBATCH --job-name=resnet_scales
#SBATCH --partition=ai
#SBATCH --reservation=ai
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=00:08:00
#SBATCH --output resnets_%j.out
#SBATCH --error resnets_%j.err
#SBATCH --array=0

source .venv/bin/activate
python main.py --workdir=./resnet_{$1}_imagenet_seed_$SLURM_ARRAY_TASK_ID --config=configs/v100_x8_mixed_precision.py --config.model=ResNet{$1} --config.seed=$SLURM_ARRAY_TASK_ID