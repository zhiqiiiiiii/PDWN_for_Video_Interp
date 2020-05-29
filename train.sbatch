#!/bin/bash
#
#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#
module purge
module load cuda/10.0.130
module load cudnn/10.0v7.4.2.24
source ~/.bashrc
#conda activate myenv

#echo "myenv activated"

python train.py --dataset vimeo_tri --context True --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1