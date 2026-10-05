#!/bin/bash

#SBATCH --time=1:00:00 
#SBATCH --job-name=AttentiveMOS
#SBATCH --account=PAS3309

#SBATCH --mem=64gb
#SBATCH -o evaluation/seeds/999/test_log.out

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1

module load miniconda3/24.1.2-py310
module load cuda/12.4.1 
source activate env
python data_modeling/test.py