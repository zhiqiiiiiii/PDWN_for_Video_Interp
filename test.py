import os
import torch
import numpy as np
from time import time
from torch.utils.data.dataset import Dataset
import torchvision
import sys
import os
from skimage.measure import compare_ssim, compare_mse, compare_psnr
import random
#from util.vis_offset import flow_to_image
from PIL import Image
import torch.nn.functional as F

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
parser.add_argument('--dataset', type=str, default='vimeo_tri', help='vimeo_tri|vimeo_sev|UCF')
parser.add_argument('--image_size', nargs='+', type=int, default=(64,64))
parser.add_argument('--num_input_frames', type=int, default=2)
parser.add_argument('--num_output_frames', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='refine', help='refine|refine_4')
parser.add_argument('--checkpoint_path', type=str, default='/scratch/zc1337/deform_dfn/checkpoints')
parser.add_argument('--result_path', type=str, default='/scratch/zc1337/deform_dfn/results')
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
parser.add_argument('--save_img', type=bool, default=False, help='whether to save images')
parser.add_argument('--save_freq', type=int, default=10, help='frequency to save images')
parser.add_argument('--visualize_offset', type=bool, default=False, help='whether to visualize offset')
parser.add_argument('--interpolation', type=bool, default=True)
parser.add_argument('--DVF', type=bool, default=False)
parser.add_argument('--deep', type=bool, default=False)
parser.add_argument('--context', type=bool, default=False)
parser.add_argument('--resize', type=bool, default=False)
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

    
if param.dataset == 'UCF':
    from data.UCF_dataset_test import VideoDataset
    dataset = VideoDataset(num_input_frames=2, num_output_frames=1, 
                               interpolation=param.interpolation, DVF=True)
    in_ch = 3
elif param.dataset == 'vimeo_tri': 
    from data.Vimeo_dataset_tri import VideoDataset
    dataset = VideoDataset(split='test', interpolation=param.interpolation, crop=False, flip=False, reverse=False)
    in_ch = 3
elif param.dataset == 'vimeo_sev':
    from data.Vimeo_dataset_7 import VideoDataset
    dataset = VideoDataset(split='test', interpolation=param.interpolation, crop=False, flip=False, reverse=False, negative=False)
    in_ch = 3
else:
    print('Please use the correct dataset name')
    sys.exit()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size, num_workers=0, shuffle=False)
print("dataset loaded")


if param.num_input_frames == 2:
    from models.refine import UNet
    net = UNet(in_ch, image_size=param.image_size, 
             num_input_frames=param.num_input_frames, 
             interpolation=param.interpolation, context=param.context)
else:
    from models.refine_4 import UNet
    net = UNet(in_ch, image_size=param.image_size, 
                 num_input_frames=param.num_input_frames, 
                 interpolation=param.interpolation, context=param.context)
    
if param.use_cuda and torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda')
    net = net.cuda()
    
para = sum(p.numel() for p in net.parameters())
print('Model {} : params: {:1f}M'.format(net._get_name(), para))

load_path = '%s_%s.pth' % (param.name, param.model_load)
load_path = os.path.join(check_dir, load_path)
state_dict = torch.load(load_path, map_location=device)
model_dict = net.state_dict()
state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
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
        if param.num_input_frames == 2:
            if param.dataset == 'vimeo_tri':
                input_frames = torch.cat([data[:,:,0:1,:,:].clone(), 
                                              data[:,:,2:3,:,:].clone()], dim=2)
                gt = data[:,:,1,:,:].clone()
            elif param.dataset == 'UCF':
                input_frames = data[:,:,0:2,:,:].clone()
                gt = data[:,:,2,:,:].clone()
            elif param.dataset == 'vimeo_sev':
                input_frames = torch.cat([data[:,:,1:2,:,:].clone(), 
                                              data[:,:,3:4,:,:].clone()], dim=2)
                gt = data[:,:,2,:,:].clone()
        else:
            input_frames = torch.cat([data[:,:,0:2,:,:].clone(), 
                                      data[:,:,3:5,:,:].clone()], dim=2)
            gt = data[:,:,2,:,:].clone()

        if param.use_cuda and torch.cuda.is_available():
            input_frames = input_frames.cuda()
            gt = gt.cuda()

        
        predictions, offset, weights, _,_ = net(input_frames, 
                           num_input_frames=param.num_input_frames,
                           num_output_frames=param.num_output_frames,
                           use_cuda=param.use_cuda, 
                           interpolation=param.interpolation)
            
            
        img_pred = predictions[0,:,:,:]
#         img_pred = ((img_pred.detach().cpu().numpy() + 1.0) / 2.0 * 255.0
         
        img_pred = np.clip(img_pred.detach().cpu().numpy() * 255.0, 0.0, 255.0)
        img_true = gt[0,:,:,:]
#         img_true = ((img_true.detach().cpu().numpy() + 1.0) / 2.0 * 255.0)
        img_true = np.clip(img_true.detach().cpu().numpy() * 255.0, 0.0, 255.0)


        img_pred = np.transpose(img_pred, (1,2,0))
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
                img_true = unloader(img_true.astype(np.uint8))
                img_filename = ('%s_true.png' % (i))
                path = os.path.join(save_dir, img_filename)
                img_true.save(path) 


        if param.save_img:
            if i % param.save_freq == 0:
                if not param.interpolation:
                    for k in range(param.num_input_frames):
                        img = data[k][0,:,:,:]
                        img = img.detach().cpu().numpy()
                        img = np.transpose(img, (1,2,0))
                        img = Image.fromarray((img*255.0).astype(np.uint8))
                        img_filename = ('%s_input_frame_%s.png' % (i, k))
                        path = os.path.join(save_dir, img_filename)
                        img.save(path)
                else:
                    for k in range(param.num_input_frames):
                        img = input_frames[0,:,k,:,:]
                        torch2im(img, os.path.join(save_dir, ('%s_input_frame_%s.png' % (i, k))))
                        
                from util.util import flow_to_color
                weights = weights.cpu().detach().numpy()
                h = offset.shape[2]
                w = offset.shape[3]
                w_frames = abs(weights[0,0,...] - weights[0,1,...])*255.0
                weight_map = unloader(w_frames.astype(np.uint8))
                path = os.path.join(save_dir, ('%s_weight.png' % (i)))
                weight_map.save(path)
                offset = offset.cpu().detach().numpy()
                xy = offset[0,...]
                f, h, w = xy.shape
                xy[0,...] = xy[0,...]/w
                xy[1,...] = xy[1,...]/h
                xy[2,...] = xy[2,...]/w
                xy[3,...] = xy[3,...]/h

                flow_map = flow_to_color(np.transpose(xy[[0,1],...],(1,2,0)))
                flow_map = unloader(flow_map.astype(np.uint8))
                img_filename = ('%s_flow_0_%s.png' % (i, 0))
                path = os.path.join(save_dir, img_filename)
                flow_map.save(path)

                flow_map = flow_to_color(np.transpose(xy[[2,3],...],(1,2,0)))
                flow_map = unloader(flow_map.astype(np.uint8))
                img_filename = ('%s_flow_1_%s.png' % (i, 1))
                path = os.path.join(save_dir, img_filename)
                flow_map.save(path)


for j in range(param.num_output_frames):
    message = 'Frame %s, Average mse: %.5f, Average psnr: %.5f, Average ssim: %.5f' % (j, np.mean(mses[:,j]), np.mean(psnrs[:,j]), np.mean(ssims[:,j]))
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)