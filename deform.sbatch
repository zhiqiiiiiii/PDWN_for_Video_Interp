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

python finetune_sintel.py --dataset sintel --name Sintel_refine_ft --residual True --learning_rate 0.00001 --milestone 5 15 25 35 45 --decay 0.2 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 4 --use_cuda True --display_freq 20 --save_freq 200 --epoch_save_freq 1

#python finetune_rdcrop.py --dataset UCF --name UCF_refine_ft --residual True --learning_rate 0.000001 --milestone 10 20 30 40 50 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python finetune_whole.py --dataset vimeo_tri --name vimeo_refine_ftwhole --residual True --learning_rate 0.0001 --milestone 5 15 25 35 45 55 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python finetune.py --dataset vimeo_tri --name vimeo_refine_ftres --residual True --learning_rate 0.0001 --milestone 5 15 25 35 45 55 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_sev --name vimeo_refine_4_resdeep --residual True --deep True --continue_train True --milestone 20 30 40 50 --learning_rate 0.0001 --decay 0.5 --interpolation True --num_input_frame 4 --num_output_frame 1  --loss L1 --batch_size 12 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_jointocc_resdeep --model jointocc --residual True --deep True --learning_rate 0.0001 --milestone 30 40 50 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_residual_deep --residual True --deep True --continue_train True --model_load 12 --epoch_start 13 --learning_rate 0.0001 --milestone 10 20 30 40 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1


#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_upocc_resdeep --model upocc --residual True --deep True --continue_train True --epoch_start 1 --learning_rate 0.0001 --milestone 29 39 49 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_7 --model refine_7 --continue_train True --learning_rate 0.0001 --milestone 30 40 50 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 32 --use_cuda True --display_freq 50 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_upocc --model upocc --learning_rate 0.0001 --milestone 30 40 50 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_jointocc --model jointocc --learning_rate 0.0001 --milestone 30 40 50 --decay 0.5 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_res --model res --learning_rate 0.0001 --continue_train True --milestone 30 40 50 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_l --model refine_l --continue_train True --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine --model_load 30 --continue_train True --epoch_start 31 --learning_rate 0.0001 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 50 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_residual --residual True --epoch_start 29 --milestone 2 7 12 17 --decay 0.5 --continue_train True --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 32 --use_cuda True --display_freq 200 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_context --context True --continue_train True --epoch_start 24 --interpolation True --num_input_frame 2 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 50 --save_freq 2000 --epoch_save_freq 1

#python train_refine_new.py --dataset vimeo_tri --name vimeo_refine_4 --model_load 32 --epoch_start 32 --milestone 3 10 15 20 --continue_train True --interpolation True --num_input_frame 4 --num_output_frame 1  --loss L1 --batch_size 16 --use_cuda True --display_freq 50 --save_freq 2000 --epoch_save_freq 2

#python train_refine.py --dataset adobe --name adobe_2_refine --interpolation True --learning_rate 0.0001 --num_input_frame 2 --num_output_frame 1 --dy_filter_size 2 --loss L1 --epochs 350 --image_size 352 352 --batch_size 12 --use_cuda True --display_freq 50 --save_freq 2000

#python train_new.py --dataset all --name all_2_no_grid_only_one_warp_regu --warp_regu True --lamda_warp 0.05 --only_one True --mode st --model unet_slowmo_nogrid --interpolation True --learning_rate 0.0001 --num_input_frame 4 --num_output_frame 1 --dy_filter_size 2 --loss L1 --epochs 350 --image_size 352 352 --batch_size 32 --use_cuda True --display_freq 50 --save_freq 5000

#python train_new.py --dataset all --name all_2_no_grid_only_one --continue_train True --model_load 108 --epoch_start 100 --only_one True --mode st --model unet_slowmo_nogrid --interpolation True --learning_rate 0.0001 --num_input_frame 4 --num_output_frame 1 --dy_filter_size 2 --loss L1 --epochs 350 --image_size 352 352 --batch_size 32 --use_cuda True --display_freq 50 --save_freq 5000

#python train_new.py --dataset adobe --name adobe_2_slowmo_nogrid --mode st --model unet_slowmo_nogrid --continue_train True --interpolation True --learning_rate 0.0001 --num_input_frame 2 --num_output_frame 1 --dy_filter_size 2 --loss L1 --epochs 350 --image_size 352 352 --batch_size 32 --use_cuda True --display_freq 50 --save_freq 1500 --epoch_save_freq 2

#python train_new.py --dataset all --name all_3_slowmo_only_one --only_one True --mode st --model unet_slowmo --interpolation True --learning_rate 0.0001 --num_input_frame 4 --num_output_frame 1 --dy_filter_size 3 --loss L1 --epochs 350 --image_size 352 352 --batch_size 32 --use_cuda True --display_freq 50 --save_freq 2000 --epoch_save_freq 2

#python train_new.py --dataset adobe --name adobe_3_slowmo_new --mode st --model unet_slowmo_new --interpolation True --learning_rate 0.0001 --num_input_frame 2 --num_output_frame 1 --dy_filter_size 3 --loss L1 --epochs 350 --image_size 352 352 --batch_size 32 --use_cuda True --display_freq 50 --save_freq 1000 --epoch_save_freq 2
