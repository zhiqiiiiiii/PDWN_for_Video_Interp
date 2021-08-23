import re
import argparse
import os
import torch
import cv2
import torchvision.transforms as transforms
from skimage.measure import compare_psnr
from PIL import Image
from tqdm import tqdm
import numpy as np
import math

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def torch2im(tensor_gpu, path):
    unloader = torchvision.transforms.ToPILImage()
#     img = (tensor_gpu.detach().cpu() + 1.0) / 2.0
    img = tensor_gpu.detach().cpu()
    img = unloader(img)
    img.save(path)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_input_frames', type=int, default=2)
parser.add_argument('--num_output_frames', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='deform', help='deform|refine')
parser.add_argument('--kernel', type=int, default=1, help='frequency to save results')
parser.add_argument('--groups', type=int, default=1, help='frequency to save results')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
parser.add_argument('--context', type=bool, default=False)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--t_interp', type=int, default=4, help='times of interpolating')
parser.add_argument('--fps', type=int, default=-1, help='specify the fps.')
parser.add_argument('--slow_motion', action='store_true', help='using this flag if you want to slow down the video and maintain fps.')
param = parser.parse_args()

check_dir = os.path.join(param.checkpoint_path, param.name)
mkdir('./demo')

from models.unet_deformv2_plus import UNet
net = UNet(in_ch, context=param.context, kernel=param.kernel, groups=param.groups)
net = net.cuda()
para = sum(p.numel() for p in net.parameters())
print('Model {} : params: {:1f}M'.format(net._get_name(), para))

device = torch.device('cuda')
load_path = '%s_%s.pth' % (param.name, param.model_load)
load_path = os.path.join(check_dir, load_path)
state_dict = torch.load(load_path, map_location=device)
model_dict = net.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
print(load_path)
net.load_state_dict(state_dict)
print("model loaded")
net.eval()

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):


    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        return flipped_img.convert('RGB')
    
    
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
#normalize = transforms.Normalize(mean=mean,
#                                 std=std)
transform = transforms.Compose([transforms.ToTensor()])#, normalize])

negmean = [-1 for x in mean]
restd = [2, 2, 2]
revNormalize = transforms.Normalize(mean=negmean, std=restd)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


def ToImage(frame0, frame1):
    with torch.no_grad():
        img0 = frame0.cuda()
        img1 = frame1.cuda()
        input_frames = torch.stack([img0, img1], dim=2)
        imgt,_,_,_,_ = net(input_frames)
        imgt = torch.clamp(imgt, max=1., min=0.)
    return imgt

def IndexHelper(i, digit):
    index = str(i)
    for j in range(digit-len(str(i))):
        index = '0'+index
    return index

def VideoToSequence(path, time):
    video = cv2.VideoCapture(path)
    dir_path = 'frames_tmp'
    dir_path_re = 'frames_tmp_re'
    os.system("rm -rf %s" % dir_path)
    os.mkdir(dir_path)
    os.system("rm -rf %s" % dir_path_re)
    os.mkdir(dir_path_re)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('making ' + str(length) + ' frame sequence in ' + dir_path)
    i = -1
    while (True):
        (grabbed, frame) = video.read()
        if not grabbed:
            break
        i = i + 1
        index = IndexHelper(i*time, len(str(time*length)))
        cv2.imwrite(dir_path + '/' + index + '.png', frame)
        if i % param.t_interp == 0:
            cv2.imwrite(dir_path_re + '/' + index + '.png', frame)
       # print(index)
    print("num of frames: ", length)
    print("fps:", fps)
    output_fps = fps if param.slow_motion else fps/param.t_interp
    print(output_fps, len(os.listdir(dir_path_re)))
    output_file = "./demo/" + param.video_path.split('/')[-1].split('.')[0] + '_re.mp4'
    os.system("ffmpeg -framerate " + str(output_fps) + " -pattern_type glob -i '" + dir_path_re + "/*.png' -pix_fmt yuv420p -b:v 120000k " + output_file)
    os.system("rm -rf %s" % dir_path_re)
    return [dir_path, length, fps]

def main():
    # initial
    iter = math.log(param.t_interp, int(2))
    if iter%1:
        print('the times of interpolating must be power of 2!!')
        return
    iter = int(iter)
 
    count = 0
    [dir_path, frame_count, fps] = VideoToSequence(param.video_path, param.t_interp)

    for i in range(iter):
        print('processing iter' + str(i+1) + ', ' + str((i+1)*frame_count) + ' frames in total')
        filenames = os.listdir(dir_path)
        filenames.sort()
        for i in range(0, len(filenames) - 1):
            arguments_strFirst = os.path.join(dir_path, filenames[i])
            arguments_strSecond = os.path.join(dir_path, filenames[i + 1])
            index1 = int(re.sub("\D", "", filenames[i]))
            index2 = int(re.sub("\D", "", filenames[i + 1]))
            index = int((index1 + index2) / 2)
            arguments_strOut = os.path.join(dir_path,
                                            IndexHelper(index, len(str(param.t_interp * frame_count))) + ".png")
            #print(arguments_strOut)
            X0 = transform(_pil_loader(arguments_strFirst)).unsqueeze(0)
            X1 = transform(_pil_loader(arguments_strSecond)).unsqueeze(0)
            assert (X0.size(2) == X1.size(2))
            assert (X0.size(3) == X1.size(3))

            intWidth = X0.size(3)
            intHeight = X0.size(2)
            channel = X0.size(1)
            if not channel == 3:
                print('Not RGB image')
                continue
            count += 1

            first, second = X0, X1
            imgt = ToImage(first, second)
            #print(imgt.shape)

            imgt_np = imgt.squeeze(0).cpu().numpy()
            imgt_png = np.uint8(imgt_np.transpose(1, 2, 0)[:, :, ::-1] * 255.0)
            cv2.imwrite(arguments_strOut, imgt_png)

    output_file = "./demo/" + param.video_path.split('/')[-1].split('.')[0] + '.mp4'
    if param.fps != -1:
        output_fps = param.fps
    else: 
        output_fps = fps if param.slow_motion else param.t_interp*fps
    # os.system("module load ffmpeg/intel/3.2.2")
    os.system("ffmpeg -framerate " + str(output_fps) + " -pattern_type glob -i '" + dir_path + "/*.png' -pix_fmt yuv420p -b:v 120000k " + output_file)
    os.system("rm -rf %s" % dir_path)


main()