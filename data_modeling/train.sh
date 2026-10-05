#!/bin/bash

#SBATCH --time=24:00:00 
#SBATCH --job-name=AttentiveMOS
#SBATCH --account=PAS3309

#SBATCH --mem=64gb
#SBATCH -o checkpoints/seeds/999/train_log.out

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1

module load miniconda3/24.1.2-py310
module load cuda/12.4.1
source activate env
python data_modeling/train.py