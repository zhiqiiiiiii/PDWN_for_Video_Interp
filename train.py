import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import sys
from models.networks import init_weights


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
parser.add_argument('--dataset', type=str, default='vimeo_tri', help='vimeo_tri|vimeo_sev')
parser.add_argument('--image_size', nargs='+', type=int, default=(64,64), help="h,w")
parser.add_argument('--num_input_frames', type=int, default=2)
parser.add_argument('--num_output_frames', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--model', type=str, default='refine', help='refine: 2 input|refine_4: 4 input')
parser.add_argument('--checkpoint_path', type=str, default='/scratch/zc1337/deform_dfn/checkpoints')
parser.add_argument('--result_path', type=str, default='./results')
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--epoch_start', type=int, default=0)
parser.add_argument('--model_load', type=str, default='latest', help='saved model to continue train')
parser.add_argument('--loss', type=str, default='L1', help='MSE|L1')
parser.add_argument('--display_freq', type=int, default=200, help='frequency to display result')
parser.add_argument('--save_freq', type=int, default=2000, help='frequency to save results')
parser.add_argument('--epoch_save_freq', type=int, default=2, help='frequency to save model')
parser.add_argument('--interpolation', type=bool, default=True)
parser.add_argument('--resize', type=bool, default=False)
parser.add_argument('--context', type=bool, default=False)
parser.add_argument('--crop', type=bool, default=False)
parser.add_argument('--milestone', nargs='+', type=int, default=(30, 40, 50), help="h,w")
parser.add_argument('--decay', type=float, default=0.1)
param = parser.parse_args()

device = torch.device('cpu')


# filepath to save checkpoints
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


# load dataset and model   
if param.num_input_frames == 2:
    if param.dataset == 'vimeo_sev':
        from data.Vimeo_dataset_7 import VideoDataset
        dataset = VideoDataset(split='train', interpolation=param.interpolation, crop=param.crop, flip=True, reverse=True, negative=False)
    else:
        from data.Vimeo_dataset_tri import VideoDataset
        dataset = VideoDataset(split='train', interpolation=param.interpolation, crop=param.crop, flip=True, reverse=True, negative=False)
    in_ch = 3

    from models.refine import UNet
    net = UNet(in_ch, image_size=param.image_size, num_input_frames=param.num_input_frames, context=param.context)
else:
    from data.Vimeo_dataset_7 import VideoDataset
    dataset = VideoDataset(split='train', interpolation=param.interpolation, crop=param.crop, flip=True, reverse=True, negative=False)
    in_ch = 3
    from models.refine_4 import UNet
    net = UNet(in_ch, image_size=param.image_size, 
                 num_input_frames=param.num_input_frames, context=param.context, deep=param.deep)
data_loader = DataLoader(dataset, batch_size=param.batch_size, 
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  shuffle=True)    


para = sum(p.numel() for p in net.parameters())
print('Model {} : params: {:1f}M'.format(net._get_name(), para))

if param.continue_train:
    load_path = '%s_%s.pth' % (param.name, param.model_load)
    load_path = os.path.join(save_dir, load_path)
    state_dict = torch.load(load_path, map_location=device)   
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    net.load_state_dict(state_dict)
else:
    init_weights(net, init_type='kaiming')

if param.use_cuda and torch.cuda.is_available():
    print("using cuda")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    device = torch.device('cuda')
    net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=param.learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=param.milestone, gamma=param.decay)

if param.loss == 'MSE':
    criterion = nn.MSELoss()
elif param.loss == 'L1':
    criterion = nn.L1Loss()

unloader = torchvision.transforms.ToPILImage()


iter = 0
for epoch in range(param.epoch_start, param.epoch_start+param.epochs):    
    
    for i, data in enumerate(data_loader):

        if param.num_input_frames == 2:
            if param.dataset == 'vimeo_sev':
                input_frames = torch.cat([data[:,:,1:2,:,:].clone(), 
                                          data[:,:,3:4,:,:].clone()], dim=2)
                gt = data[:,:,2,:,:].clone()
  #                     print(input_frames.shape)
  #                     print(data.shape)
            else:
                input_frames = torch.cat([data[:,:,0:1,:,:].clone(), 
                                              data[:,:,2:3,:,:].clone()], dim=2)
                gt = data[:,:,1,:,:].clone()
        if param.use_cuda and torch.cuda.is_available():
            input_frames = input_frames.cuda()
            gt = gt.cuda()

        
        optimizer.zero_grad()

        if param.deep:
            predictions, inter, _, _, x_0, x_1 = net(input_frames, 
                           num_input_frames=param.num_input_frames,
                           num_output_frames=param.num_output_frames,
                           use_cuda=param.use_cuda, 
                           interpolation=param.interpolation,
                           warp=param.warp_regu)
        else:
            predictions, _, _, x_0, x_1 = net(input_frames, 
                               num_input_frames=param.num_input_frames,
                               num_output_frames=param.num_output_frames,
                               use_cuda=param.use_cuda, 
                               interpolation=param.interpolation,
                               warp=param.warp_regu)
      
        loss = criterion(predictions, gt)
        if param.deep:
            loss = (loss + criterion(inter, gt))/2.0
                
        loss.backward()
        optimizer.step()

        
        if iter%param.display_freq == 0:
            message = 'epoch:{}, iter:{} --- loss: {} --- lr: {}'.format(epoch, i, 
                                                                loss.item(), 
                                                                optimizer.param_groups[0]["lr"])
            print(message)
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)


        if iter%param.save_freq == 0:
            img_pred = predictions[0,:,:,:]
            torch2im(img_pred, os.path.join(img_dir, ('%s_%s_pred_%s.png' % (epoch, iter, 0))))
            img_true = gt[0,:,:,:]
            torch2im(img_true, os.path.join(img_dir, ('%s_%s_true_%s.png' % (epoch, iter, 0))))
            for k in range(param.num_input_frames):
                img = input_frames[0,:,k,:,:]
                torch2im(img, os.path.join(img_dir, ('%s_%s_input_frame_%s.png' % (epoch, iter, k))))
            path = os.path.join(save_dir, ('%s_latest.pth' % param.name))
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), path)
            else:
                torch.save(net.state_dict(), path)
        iter += 1       
    

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
            
    scheduler.step()

