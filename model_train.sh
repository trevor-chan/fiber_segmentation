#!/bin/bash
#SBATCH --job-name=fibersegTrain
#SBATCH --out="slurm-%j.out"
#SBATCH --time=40:00:00
#SBATCH --nodes=1 --ntasks=4 --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --partition=gpu

module load CUDA
module load cuDNN

source activate genv

python train.py
