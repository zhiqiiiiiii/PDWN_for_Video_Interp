# PDWN: Pyramid Deformable Warping Network for Video Interpolation

## Table of Contents
* demos
* requirements
* train
* test
* evaluate on videos

## demos 

<video id="video" controls="" preload="none">
      <source id="mp4" src="https://github.com/zhiqiiiiiii/PDWN/blob/master/demos/kimono.mp4" type="video/mp4">
</video>

## Requirements
* Ubuntu
* Pytorch
* Cuda (10.1) & Cudnn (10.1)
* mmcv-full (https://github.com/open-mmlab/mmcv. please follow the guidence to install mmcv properly.)
* ffmpeg

## Train
```python
python train.py --name experiment --dataset vimeo_tri --model deform --kernel 3 --context True --loss L1 --batch_size 8 --use_cuda True
```

## Test
```python
python test.py --dataset vimeo_tri --name vimeo_deform_3_context --model deform --context True --kernel 3 --model_load 56 --kernel 3 --interpolation True --num_input_frame 2 --num_output_frame 1 --use_cuda True --save_img True --save_freq 1
```

## Evaluate on videos
```python
python seq_eval --video_path ./sunflower_1080p25.mp4 --name vimeo_deform_3_context --model deform --context True --kernel 3 --model_load 56 --t_interp 2
```
