#!/usr/bin/env bash

#SBATCH --partition=wacc
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --job-name=task1
#SBATCH --output=inference.out

module load cuda
module load clang
module purge
plaidml-setup


python3 pretrained_inference.py



