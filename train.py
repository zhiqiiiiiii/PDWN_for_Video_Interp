import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import sys
import os
from skimage.measure import compare_ssim, compare_mse, compare_psnr 
from models.networks import init_weights
from models.Charbonnier import CharbonnierLoss

torch.backends.cudnn.benchmark = False
   
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def torch2im(tensor_gpu, path):
    unloader = torchvision.transforms.ToPILImage()
#     img = (tensor_gpu.detach().cpu() + 1.0) / 2.0
    img = tensor_gpu.detach().cpu()
    img = np.clip(img, 0.0, 1.0)
    img = unloader(img)
    img.save(path)
        
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, help='path to your saved dataset')
parser.add_argument('--dataset', type=str, default='vimeo_tri', help='mnist|highway|UCF|KTH|kitti')
parser.add_argument('--num_input_frames', type=int, default=2)
parser.add_argument('--num_output_frames', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='deform_dfn', help='deform_dfn')
parser.add_argument('--kernel', type=int, default=3, help='kernel size for deform conv')
parser.add_argument('--groups', type=int, default=1, help='groups size for deform conv')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
parser.add_argument('--loss', type=str, default='L1', help='MSE|L1')
parser.add_argument('--display_freq', type=int, default=200, help='frequency to display result')
parser.add_argument('--save_freq', type=int, default=2000, help='frequency to save results')
parser.add_argument('--epoch_save_freq', type=int, default=2, help='frequency to save model')
parser.add_argument('--resize', type=bool, default=False)
parser.add_argument('--context', type=bool, default=False)
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--milestone', nargs='+', type=int, default=(30, 50, 70), help="h,w")
parser.add_argument('--decay', type=float, default=0.5)
param = parser.parse_args()
device = torch.device('cpu')

# save checkpoints
check_dir = param.checkpoint_path
save_dir = os.path.join(check_dir, param.name)
img_dir = os.path.join(save_dir, 'imgs')
mkdir(check_dir)
mkdir(save_dir)
mkdir(img_dir)

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

# load dataset
from data.Vimeo_dataset_tri import VideoDataset
dataset = VideoDataset(data_root=param.dataroot, split='train', crop=param.crop, flip=True, reverse=True, 
                       negative=False, normalize=False)
data_loader = DataLoader(dataset, batch_size=param.batch_size, num_workers=4, pin_memory=True, 
                         drop_last=True,shuffle=True)    
in_ch = 3

# load model
from models.unet_deformv2_plus import UNet
net = UNet(in_ch, context=param.context, kernel=param.kernel, groups=param.groups)
para = sum(p.numel() for p in net.parameters())
print('Model {} : params: {}'.format(net._get_name(), para))

# prepare evaluation
from data.Middle_dataset import VideoDataset
dataset_eval = VideoDataset(split='other', num_input_frames=param.num_input_frames, normalize=False)
data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=1, num_workers=0, shuffle=False)
print("evaluation dataset loaded")
criterionMSE = torch.nn.MSELoss()
mses = np.zeros((len(dataset_eval), param.num_output_frames))
psnrs = np.zeros((len(dataset_eval), param.num_output_frames))
ssims = np.zeros((len(dataset_eval), param.num_output_frames))

# resume training
if param.continue_train:
    load_path = '%s_%s.pth' % (param.name, param.model_load)
    load_path = os.path.join(save_dir, load_path)
    state_dict = torch.load(load_path, map_location=device)   
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    net.load_state_dict(state_dict)
    
# use cuda
if param.use_cuda and torch.cuda.is_available():
    print("using cuda")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    device = torch.device('cuda')
    net = net.cuda()

# initializa optimizer
optimizer = optim.Adam(net.parameters(), lr=param.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=param.milestone, gamma=param.decay)

criterion = nn.L1Loss()

unloader = torchvision.transforms.ToPILImage()
iter = 0
for epoch in range(param.epoch_start, param.epoch_start+param.epochs):    
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        
        input_frames = torch.cat([data[:,:,0:1,:,:].clone(), 
                                      data[:,:,2:3,:,:].clone()], dim=2)
        input_frames = input_frames.cuda()
        gt = data[:,:,1,:,:].clone().cuda()

        
        predictions, offsets, weights, _, masks = net(input_frames, use_cuda=param.use_cuda, channel_wise_norm=True)
        loss = criterion(predictions, gt)

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()

        if iter%param.display_freq == 0:
            message = 'epoch:{}, iter:{} - loss:{:.5} '.format(epoch, i, loss.item())
            message += '-- lr: {:.5}'.format(optimizer.param_groups[0]["lr"])
            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

        if iter%param.save_freq == 0:
            net.eval()
            with torch.no_grad():
                total_mse, total_psnr, total_ssim = 0, 0, 0
                for i, data in enumerate(data_loader_eval):
                    input_frames = torch.cat([data[:,:,0:1,:,:].clone(), 
                                                      data[:,:,2:3,:,:].clone()], dim=2)
                    input_frames = input_frames.cuda()
                    gt = data[:,:,1,:,:].clone().cuda()

                    prediction, _, _, _,_ = net(input_frames, use_cuda=param.use_cuda, channel_wise_norm=True)
                    img_pred = prediction[0,:,:,:].detach().cpu().numpy()
                    img_pred = np.transpose(np.clip(img_pred, 0.0, 255.0), (1,2,0))
                    img_true = gt[0,:,:,:].detach().cpu().numpy()
                    img_true = np.clip(np.transpose(img_true, (1,2,0)), 0.0, 255.0)

                    mse = compare_mse(img_pred.astype(np.uint8), img_true.astype(np.uint8))
                    psnr = compare_psnr(img_true.astype(np.uint8), img_pred.astype(np.uint8))
                    ssim = compare_ssim(img_pred.astype(np.uint8), img_true.astype(np.uint8), multichannel=True)

                    total_mse += mse
                    total_psnr += psnr
                    total_ssim += ssim
                message = '=== Evaluation, average mse: %.3f, average psnr: %.3f, average ssim: %.3f ===' % (
                               total_mse/len(dataset_eval), total_psnr/len(dataset_eval), total_ssim/len(dataset_eval))
                print(message)
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
            net.train()

            path = os.path.join(save_dir, ('%s_latest.pth' % param.name))
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), path)
            else:
                torch.save(net.state_dict(), path)
        iter += 1       
    scheduler.step()

    if epoch % param.epoch_save_freq == 0:
        message = 'saving the model at the end of epoch %d, total iters %d' % (epoch, iter)
        print(message)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)
        path = os.path.join(save_dir, ('%s_%s.pth' % (param.name, epoch)))
        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), path)
        else:
            torch.save(net.state_dict(), path)