#!/bin/bash

#SBATCH --job-name=resnet_data
#SBATCH --partition=ai,ainlp
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=10M
#SBATCH --time=00:5:00
#SBATCH --output resnets_%j.out
#SBATCH --error resnets_%j.err
#SBATCH --array=0-4

module load GCC/10.3.0
module load CMake/3.20.1

source .venv/bin/activate
python main.py --workdir=./resnet_{$1}_imagenet_seed_$SLURM_ARRAY_TASK_ID --config=configs/default.py --config.model=ResNet{$1} --config.seed=$SLURM_ARRAY_TASK_ID