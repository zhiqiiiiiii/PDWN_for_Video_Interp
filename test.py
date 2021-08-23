import os
import torch
import numpy as np
from time import time
from torch.utils.data import Dataset, DataLoader
import torchvision
import sys
import os
from skimage.measure import compare_ssim, compare_mse, compare_psnr
import random
from util.vis_offset import flow_to_image
from PIL import Image
import torch.nn.functional as F
import cv2

import math
netNetwork = None

def estimate(tenFirst, tenSecond):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda()
        for param in netNetwork.parameters():
            param.requires_grad = False
        netNetwork.eval()

    assert(tenFirst.shape[2] == tenSecond.shape[2])
    assert(tenFirst.shape[3] == tenSecond.shape[3])

    intWidth = tenFirst.shape[3]
    intHeight = tenFirst.shape[2]

    tenPreprocessedFirst = tenFirst
    tenPreprocessedSecond = tenSecond

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0)) #1024
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0)) #448

    tenPreprocessedFirst = F.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedSecond = F.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    
    tenFlow = netNetwork(tenPreprocessedFirst, tenPreprocessedSecond)
    
    tenFlow = 20.0 * torch.nn.functional.interpolate(input=tenFlow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
    return tenFlow


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def torch2im(tensor_gpu, path):
    unloader = torchvision.transforms.ToPILImage()
    img = tensor_gpu.detach().cpu()
    img = unloader(img)
    img.save(path)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, help='path to your saved dataset')
parser.add_argument('--dataset', type=str, default='vimeo_tri')
parser.add_argument('--num_input_frames', type=int, default=2)
parser.add_argument('--num_output_frames', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='refine', help='deform_dfn|dfn')
parser.add_argument('--kernel', type=int, default=1, help='kernel size of deform conv')
parser.add_argument('--groups', type=int, default=1, help='group size of deform conv')
parser.add_argument('--checkpoint_path', type=str, default='/scratch/zc1337/deform_dfn/checkpoints')
parser.add_argument('--result_path', type=str, default='/scratch/zc1337/deform_dfn/results')
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
parser.add_argument('--save_img', type=bool, default=False, help='whether to save images')
parser.add_argument('--save_freq', type=int, default=10, help='frequency to save images')
parser.add_argument('--visualize_offset', type=bool, default=False, help='whether to visualize offset')
parser.add_argument('--interpolation', type=bool, default=True)
parser.add_argument('--context', type=bool, default=False)
parser.add_argument('--resize', type=bool, default=False)
parser.add_argument('--ensemble', type=bool, default=False)
param = parser.parse_args()
device = torch.device('cpu')

# save checkpoints
mkdir(param.result_path)
check_dir = os.path.join(param.checkpoint_path, param.name)
save_dir = os.path.join(param.result_path, param.name+'_'+param.dataset)
mkdir(save_dir)

# save log file
log_name = os.path.join(save_dir, 'log.txt')
message = ''
message += '----------------- Options ---------------\n'
for k, v in sorted(vars(param).items()):
    comment = ''
    default = parser.get_default(k)
    if v != default:
        comment = '\t[default: %s]' % str(default)
    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
message += '----------------- End -------------------'
print(message)
with open(log_name, "a") as log_file:
    log_file.write('%s\n' % message)

# initialize dataset  
from data.Vimeo_dataset_tri import VideoDataset
dataset = VideoDataset(data_root=param.dataroot, split='test', crop=False, flip=False, reverse=False, negative=False, normalize=False, self_ensemble=param.ensemble)
in_ch = 3
data_loader = DataLoader(dataset, batch_size=1, 
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  shuffle=False)    

# initialize model
from models.unet_deformv2_plus import UNet
net = UNet(in_ch, context=param.context, kernel=param.kernel, groups=param.groups)
para = sum(p.numel() for p in net.parameters())
print('Model {} : params: {:1f}M'.format(net._get_name(), para))
if param.use_cuda and torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda')
    net = net.cuda()

# load model
load_path = '%s_%s.pth' % (param.name, param.model_load)
load_path = os.path.join(check_dir, load_path)
state_dict = torch.load(load_path, map_location=device)
model_dict = net.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
print(load_path)
net.load_state_dict(state_dict)
print("model loaded")


net.eval()
unloader = torchvision.transforms.ToPILImage()
criterionMSE = torch.nn.MSELoss()
mses = np.zeros((len(dataset), param.num_output_frames))
psnrs = np.zeros((len(dataset), param.num_output_frames))
ssims = np.zeros((len(dataset), param.num_output_frames))

with torch.no_grad():
    for i, data in enumerate(data_loader):
        if param.ensemble:
            [data, data_flip, data_mirror, data_reverse, data_flip_reverse,
             data_rotate_90, data_rotate_180, data_rotate_90_inverse] = data 
            input_frames = (torch.cat([data[:,:,0:1,:,:].clone(), 
                                          data[:,:,2:3,:,:].clone()], dim=2)).cuda()
            gt = data[:,:,1,:,:].clone().cuda()
            input_frames_flip = (torch.cat([data_flip[:,:,0:1,:,:].clone(), 
                                      data_flip[:,:,2:3,:,:].clone()], dim=2)).cuda()
            input_frames_mirror = (torch.cat([data_mirror[:,:,0:1,:,:].clone(), 
                                      data_mirror[:,:,2:3,:,:].clone()], dim=2)).cuda()
            input_frames_reverse = (torch.cat([data_reverse[:,:,0:1,:,:].clone(), 
                                      data_reverse[:,:,2:3,:,:].clone()], dim=2)).cuda()
            input_frames_flip_reverse = (torch.cat([data_flip_reverse[:,:,0:1,:,:].clone(), 
                                      data_flip_reverse[:,:,2:3,:,:].clone()], dim=2)).cuda()
            input_frames_rotate_90 = (torch.cat([data_rotate_90[:,:,0:1,:,:].clone(), 
                                      data_rotate_90[:,:,2:3,:,:].clone()], dim=2)).cuda()
            input_frames_rotate_180 = (torch.cat([data_rotate_180[:,:,0:1,:,:].clone(), 
                                      data_rotate_180[:,:,2:3,:,:].clone()], dim=2)).cuda()
            input_frames_rotate_90_inverse = (torch.cat([data_rotate_90_inverse[:,:,0:1,:,:].clone(), 
                                      data_rotate_90_inverse[:,:,2:3,:,:].clone()], dim=2)).cuda()
        else:
            input_frames = torch.cat([data[:,:,0:1,:,:].clone(), 
                                          data[:,:,2:3,:,:].clone()], dim=2)
            input_frames = input_frames.cuda()
            gt = data[:,:,1,:,:].clone().cuda()

        if param.ensemble:
            predictions, offsets, weights, _, masks = net(input_frames, use_cuda=param.use_cuda, channel_wise_norm=True)
            predictions_flip, offsets, weights, _, masks = net(input_frames_flip, use_cuda=param.use_cuda, 
                                                               channel_wise_norm=True)
            predictions_mirror, offsets, weights, _, masks = net(input_frames_mirror, use_cuda=param.use_cuda, 
                                                                 channel_wise_norm=True)
            predictions_reverse, offsets, weights, _, masks = net(input_frames_reverse, use_cuda=param.use_cuda, 
                                                                  channel_wise_norm=True)
            predictions_flip_reverse, offsets, weights, _, masks = net(input_frames_flip_reverse, use_cuda=param.use_cuda, 
                                                                       channel_wise_norm=True)
            predictions_rotate_90, offsets, weights, _, masks = net(input_frames_rotate_90, use_cuda=param.use_cuda,
                                                                    channel_wise_norm=True)
            predictions_rotate_180, offsets, weights, _, masks = net(input_frames_rotate_180, use_cuda=param.use_cuda, 
                                                                     channel_wise_norm=True)
            predictions_rotate_90_inverse, offsets, weights, _, masks = net(input_frames_rotate_90_inverse, 
                                                                            use_cuda=param.use_cuda, channel_wise_norm=True)
            img_pred = predictions[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_flip = predictions_flip[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_flip = cv2.flip(img_pred_flip, flipCode=1)
            img_pred_mirror = predictions_mirror[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_mirror = cv2.flip(img_pred_mirror, flipCode=0)
            img_pred_reverse = predictions_reverse[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_flip_reverse = predictions_flip_reverse[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_flip_reverse = cv2.flip(img_pred_flip_reverse, flipCode=1)
            img_pred_rotate_90 = predictions_rotate_90[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_rotate_90 = cv2.rotate(img_pred_rotate_90, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_pred_rotate_180 = predictions_rotate_180[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_rotate_180 = cv2.rotate(img_pred_rotate_180, cv2.ROTATE_180)
            img_pred_rotate_90_inverse = predictions_rotate_90_inverse[0,:,:,:].detach().cpu().numpy().transpose((1,2,0))
            img_pred_rotate_90_inverse = cv2.rotate(img_pred_rotate_90_inverse, cv2.ROTATE_90_CLOCKWISE)
            img_pred = (img_pred + img_pred_flip + img_pred_mirror + img_pred_reverse + img_pred_flip_reverse + img_pred_rotate_90 + img_pred_rotate_180 + img_pred_rotate_90_inverse)/8
            img_pred = np.clip(img_pred, 0.0, 255.0)
        else:
            predictions, offsets, weights, _, masks = net(input_frames, use_cuda=param.use_cuda, channel_wise_norm=True)
            
            img_pred = predictions[0,:,:,:]
            img_pred = np.clip(img_pred.detach().cpu().numpy(), 0.0, 255.0)
            img_pred = np.transpose(img_pred, (1,2,0))
        
        img_true = gt[0,:,:,:]
        img_true = np.clip(img_true.detach().cpu().numpy(), 0.0, 255.0)
        img_true = np.transpose(img_true, (1,2,0))

        mse = compare_mse(img_pred.astype(np.uint8), img_true.astype(np.uint8))
        psnr = compare_psnr(img_true.astype(np.uint8), img_pred.astype(np.uint8))
        if in_ch == 3:
            ssim = compare_ssim(img_pred.astype(np.uint8), img_true.astype(np.uint8), multichannel=True)
        else:
            ssim = compare_ssim(img_pred.astype(np.uint8), img_true.astype(np.uint8))
        message = 'Sample %s, predicting 0 frame, mse: %.3f, psnr: %.3f, ssim: %.3f' % (i, mse, psnr, ssim)
        print(message)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        mses[i, 0] = mse
        psnrs[i, 0] = psnr
        ssims[i, 0] = ssim

        if param.save_img:     
            if i % param.save_freq == 0:
                img_pred = unloader(img_pred.astype(np.uint8))
                img_filename = ('%s_pred.png' % (i))
                path = os.path.join(save_dir, img_filename)
                img_pred.save(path)

for j in range(param.num_output_frames):
    message = 'Frame %s, Average mse: %.5f, Average psnr: %.5f, Average ssim: %.5f' % (j, np.mean(mses[:,j]), np.mean(psnrs[:,j]), np.mean(ssims[:,j]))
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)


