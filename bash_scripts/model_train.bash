#!/bin/bash

#SBATCH --job-name=train_model_v1
#SBATCH --output=/home/hakeem.at/ondemand/dev/projects/deeplearning/Reliable-Explainable-Brain-Tumor-Segmentation/train_model_v1.out
#SBATCH --error=/home/hakeem.at/ondemand/dev/projects/deeplearning/Reliable-Explainable-Brain-Tumor-Segmentation/train_model_error_v1.out
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1

module load anaconda3/2022.05
conda init bash
source ~/.bashrc

conda activate BTSeg

# Run your Python script
python3 /home/hakeem.at/ondemand/dev/projects/deeplearning/Reliable-Explainable-Brain-Tumor-Segmentation/scripts/model_train.py