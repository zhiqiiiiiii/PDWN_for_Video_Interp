# PDWN: Pyramid Deformable Warping Network for Video Interpolation

Code for PDWN: Pyramid Deformable Warping Network for Video Interpolation (https://ieeexplore.ieee.org/document/9416770)

## Table of Contents
* demos
* requirements
* train
* test
* evaluate on videos

## demos
[![demo](https://res.cloudinary.com/marcomontalbano/image/upload/v1603745517/video_to_markdown/images/youtube--5rEO_-udbH0-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=5rEO_-udbH0 "demo")

## Requirements
* Ubuntu
* Pytorch
* Cuda (10.1) & Cudnn (10.1)
* mmcv-full (https://github.com/open-mmlab/mmcv. please follow the guidence to install mmcv properly.)
* ffmpeg

## Dataset
Vimeo-triplet can be downloaded from http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip

## Train
To train your own model, please use the following command:
```python
python train.py --name experiment --dataroot [PATH TO THE DATASET] --dataset vimeo_tri  --model deform --kernel 3 --loss L1 --batch_size 32 --use_cuda True
```

## Test
To replicate the results presented in the paper, please use the following command (Model is saved under ./checkpoints/vimeo_plus_single_no_norm_crop)
```python
python test.py --name vimeo_plus_single_no_norm_crop --dataroot [PATH TO THE DATASET] --ensemble True --kernel 3 --model_load latest --result_path ./results --checkpoint_path ./checkpoints --dataset vimeo_tri
```

## Evaluate on videos
```python
python seq_eval.py --video_path ./sunflower_1080p25.mp4 --name vimeo_plus_single_no_norm_crop --model deform --kernel 3 --t_interp 2
```
