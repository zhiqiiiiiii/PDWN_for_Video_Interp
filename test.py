import os
import sys
import cv2
import numpy as np
from time import time
import torch
from torch.utils.data.dataset import Dataset
import torchvision
from skimage.measure import compare_ssim, compare_mse, compare_psnr
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
parser.add_argument('--dataset', type=str, default='vimeo_tri', help='vimeo_tri|vimeo_sev|DVF')
parser.add_argument('--num_input_frames', type=int, default=2)
parser.add_argument('--num_output_frames', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='refine', help='deform_dfn|dfn')
parser.add_argument('--kernel', type=int, default=1, help='deformable convolution kernel size')
parser.add_argument('--groups', type=int, default=1, help='deformable convolution groups')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
parser.add_argument('--save_img', type=bool, default=False, help='whether to save images')
parser.add_argument('--save_freq', type=int, default=10, help='frequency to save images')
parser.add_argument('--visualize_offset', type=bool, default=False, help='whether to visualize offset')
parser.add_argument('--interpolation', type=bool, default=True)
parser.add_argument('--beyondMSE', type=bool, default=False)
parser.add_argument('--DVF', type=bool, default=False)
parser.add_argument('--deep', type=bool, default=False)
parser.add_argument('--context', type=bool, default=False)
parser.add_argument('--resize', type=bool, default=False)
parser.add_argument('--image_size', nargs='+', type=int, default=(64, 64), help="h,w")
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

    
if param.dataset == 'vimeo_tri': 
    from data.Vimeo_dataset_tri import VideoDataset
    dataset = VideoDataset(split='test', interpolation=param.interpolation, crop=False, flip=False, reverse=False)
    in_ch = 3
elif param.dataset == 'vimeo_sev':
    from data.Vimeo_dataset_7 import VideoDataset
    dataset = VideoDataset(split='test', interpolation=param.interpolation, crop=False, flip=False, reverse=False, negative=False)
    in_ch = 3
elif param.dataset == 'DVF':
    from data.UCF_dataset_test import VideoDataset
    dataset = VideoDataset(num_input_frames=2, num_output_frames=1, 
                               interpolation=param.interpolation, DVF=True)
    in_ch = 3
else:
    print('Please use the correct dataset name')
    sys.exit()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size, num_workers=0, shuffle=False)
print("dataset loaded")


if param.num_input_frames == 2:
    if param.model == 'deform':
        from models.refine_deform import UNet
        net = UNet(in_ch, context=param.context, kernel=param.kernel, groups=param.groups)
    else:
        from models.refine import UNet
        net = UNet(in_ch, deep=param.deep, residual=param.context)
else:
    from models.refine_deform_4 import UNet
    net = UNet(in_ch, context=param.context, kernel=param.kernel, groups=param.groups)
    
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
        if param.num_input_frames == 2:
            if param.dataset == 'vimeo_tri':
                input_frames = torch.cat([data[:,:,0:1,:,:].clone(), 
                                              data[:,:,2:3,:,:].clone()], dim=2)
                input_frames = input_frames.cuda()
                gt = data[:,:,1,:,:].clone().cuda()
            elif param.dataset == 'DVF':
                input_frames = data[:,:,0:2,:,:].clone().cuda()
                gt = data[:,:,2,:,:].clone().cuda()
            elif param.dataset == 'vimeo_sev':
                input_frames = torch.cat([data[:,:,1:2,:,:].clone(), 
                                              data[:,:,3:4,:,:].clone()], dim=2)
                input_frames = input_frames.cuda()
                gt = data[:,:,2,:,:].clone().cuda()
        else:
            input_frames = torch.cat([data[:,:,0:2,:,:].clone(), 
                                      data[:,:,3:5,:,:].clone()], dim=2)
            input_frames = input_frames.cuda()
            gt = data[:,:,2,:,:].clone().cuda()
            

        predictions, offset, weights, res,_ = net(input_frames, use_cuda=param.use_cuda)
            
            
        img_pred = predictions[0,:,:,:]         
        img_pred = np.clip(img_pred.detach().cpu().numpy() * 255.0, 0.0, 255.0)
        img_true = gt[0,:,:,:]
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
                
                if param.visualize_offset:
                    if res != None:
                        res = (res[0,...] - res.min())/(res.max()-res.min())
                        res = np.clip((abs(res.detach().cpu().numpy()) + 1)/2* 255.0, 0.0, 255.0)
                        res = res.astype(np.uint8)
                        res = np.transpose(res, (1,2,0))
                        res = unloader(res)
                        res.save(os.path.join(save_dir, ('%s_res.png' % (i))))
                    
                    w_frames = weights[0,0,0,...].detach().cpu().numpy()*255
                    weight_map = unloader(w_frames.astype(np.uint8))
                    path = os.path.join(save_dir, ('%s_weight.png' % (i)))
                    weight_map.save(path)
                               
                    offset_h = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
                    offset_w = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
                    """A Deformable Conv Encapsulation that acts as normal Conv layers.
                    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
                    The spatial arrangement is like:
                    .. code:: text
                        (x0, y0) (x1, y1) (x2, y2)
                        (x3, y3) (x4, y4) (x5, y5)
                        (x6, y6) (x7, y7) (x8, y8)
                        """
                    offset_0= offset[0,:2*param.kernel**2, ...].detach().cpu().numpy()
                    offset_1= offset[0,2*param.kernel**2:, ...].detach().cpu().numpy()
                    _,_,H,W = offset.shape
                    
                    img0 = input_frames[0,:,0,:,:].detach().cpu().numpy() * 255.0
                    img1 = input_frames[0,:,1,:,:].detach().cpu().numpy() * 255.0
                    img0 = np.transpose(img0, (1,2,0))
                    img1 = np.transpose(img1, (1,2,0))
                    img0 = unloader(img0.astype(np.uint8))
                    img1 = unloader(img1.astype(np.uint8))

                    sample0 = Image.new(img_pred.mode, (W*2, H))
                    sample0.paste(img0, box=(0, 0))
                    sample0.paste(img_pred, box=(W, 0))
                    sample0 = np.array(sample0)
                    
                    sample1 = Image.new(img_pred.mode, (W*2, H))
                    sample1.paste(img_pred, box=(0, 0))
                    sample1.paste(img1, box=(W, 0))
                    sample1 = np.array(sample1)
                    
                    # print(W,H)
                    for t in range(40000, H*W-4000, 11111):
                        h = t//W
                        w = t//H
                        # print(w, h)
                        for pt in range(param.kernel**2):
                            lines0_begin = (W+w, h)
                            lines0_end = (np.clip(int(w+offset_w[pt]+offset_0[2*pt, h, w]), 0, W),
                                          np.clip(int(h+offset_h[pt]+offset_0[2*pt+1, h, w]), 0, H))

                            lines1_begin = (w, h)
                            lines1_end = (W+np.clip(int(w+offset_w[pt]+offset_1[2*pt, h, w]), 0, W),
                                         np.clip(int(h+offset_h[pt]+offset_1[2*pt+1, h, w]), 0, H))
                            cv2.line(sample0,lines0_begin,lines0_end,(255,0,0))           
                            cv2.line(sample1,lines1_begin,lines1_end,(255,0,0)) 
                    
                    sample0 = unloader(sample0.astype(np.uint8))
                    sample0.save(os.path.join(save_dir, ('%s_sampling_0.png' % (i))))
                    sample1 = unloader(sample1.astype(np.uint8))
                    sample1.save(os.path.join(save_dir, ('%s_sampling_1.png' % (i))))


for j in range(param.num_output_frames):
    message = 'Frame %s, Average mse: %.5f, Average psnr: %.5f, Average ssim: %.5f' % (j, np.mean(mses[:,j]), np.mean(psnrs[:,j]), np.mean(ssims[:,j]))
    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)