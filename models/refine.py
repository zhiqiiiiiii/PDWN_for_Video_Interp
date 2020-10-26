import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import correlation
from . import ResBlock

class down(nn.Module):

    def __init__(self, inChannels, outChannels, filterSize):

        super(down, self).__init__()
        self.conv1 = nn.Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
           
    def forward(self, x):
        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        return x
    
    
class up(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
#         self.upconv = nn.ConvTranspose3d(inChannels, outChannels, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3, stride=1, padding=1)
           
    def forward(self, x, H, W):
        x = F.leaky_relu(self.conv1(x), negative_slope = 0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope = 0.1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = H - x.size()[2]
        diffX = W - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2], mode='reflect')
        return x



class UNet(nn.Module):

    def __init__(self, in_ch, num_input_frames=2,
                 num_output_frames=1, interpolation=True, context=False):
        super(UNet, self).__init__()
        # Initialize neural network blocks.
#         self.image_size = image_size
        self.in_ch = in_ch
        self.context = context
        
        self.offset_channels = 2*2
        self.weight_channels = 2

        self.ex1 = nn.Sequential(
                    nn.Conv2d(self.in_ch*num_input_frames//2, 16, kernel_size=7, stride=1, padding=3),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    )
        self.ex2 = down(16, 32, 3)
        self.ex3 = down(32, 64, 3)
        self.ex4 = down(64, 96, 3)
        self.ex5 = down(96, 128, 3)
        self.ex6 = down(128, 256, 3)
         
        self.up1   = up(256+256+81, 256)
        self.up2   = up(256+128+128+81+4, 196)
        self.up3   = up(196+96+96+81+4, 128)
        self.up4   = up(128+64+64+81+4, 96)
        self.up5   = up(96+32+32+81+4, 64)
        self.up6   = nn.Sequential(
                     nn.Conv2d(in_channels=64+16+16+81+4, out_channels=64, kernel_size=3, stride=1, padding=1),
                     nn.LeakyReLU(inplace=False, negative_slope=0.1),
                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                     nn.LeakyReLU(inplace=False, negative_slope=0.1)
                     )
        
                
        self.offset4 = nn.Conv2d(256, self.offset_channels, 3, stride=1, padding=1)
        self.offset3 = nn.Conv2d(196, self.offset_channels, 3, stride=1, padding=1)
        self.offset2 = nn.Conv2d(128, self.offset_channels, 3, stride=1, padding=1)
        self.offset1 = nn.Conv2d(96, self.offset_channels, 3, stride=1, padding=1)
        self.offset0 = nn.Conv2d(64, self.offset_channels, 3, stride=1, padding=1)
        self.offset  = nn.Conv2d(64, self.offset_channels, 3, stride=1, padding=1)

        self.blend = nn.Sequential(
                nn.Conv2d(in_channels=3 + 3 + 16*2, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
                nn.Softmax(dim=1))
         
        if self.context:
            self.enhance = ResBlock.__dict__['MultipleBasicBlock_4'](3 + 3 + 16*2, 128)
    
    
    def forward(self, input_frames, num_input_frames=2, num_output_frames=1, use_cuda=False,):
        b, c, t, h, w = input_frames.shape
        I0 = input_frames[:,:, 0, :, :].clone()
        I1 = input_frames[:,:, 1, :, :].clone()
        
        feat0_0 = self.ex1(I0)
        feat1_0 = self.ex2(feat0_0)
        feat2_0 = self.ex3(feat1_0)
        feat3_0 = self.ex4(feat2_0)
        feat4_0 = self.ex5(feat3_0)
        feat5_0 = self.ex6(feat4_0)
        
        feat0_1 = self.ex1(I1)
        feat1_1 = self.ex2(feat0_1)
        feat2_1 = self.ex3(feat1_1)
        feat3_1 = self.ex4(feat2_1)
        feat4_1 = self.ex5(feat3_1)
        feat5_1 = self.ex6(feat4_1)  
        
        volume = F.leaky_relu(correlation.FunctionCorrelation(feat5_0, feat5_1), 0.1, False)
        _, _, H, W = feat4_0.shape
        feat = self.up1(torch.cat([feat5_0, feat5_1, volume], dim=1), H, W)
        offset = self.offset4(feat)
        
                        
        feat4_0_warped = self._resample(offset[:,:2, ...], feat4_0)
        feat4_1_warped = self._resample(offset[:,2:, ...], feat4_1)
        volume = F.leaky_relu(correlation.FunctionCorrelation(feat4_0_warped, feat4_1_warped), 0.1, False)
        
        _, _, H, W = feat3_0.shape 
        feat = self.up2(torch.cat([feat4_0, feat4_1, volume, feat, offset], dim=1), H, W)
        offset = self.offset3(feat)
        
                        
        feat3_0_warped = self._resample(offset[:,:2, ...], feat3_0)
        feat3_1_warped = self._resample(offset[:,2:, ...], feat3_1)
        volume = F.leaky_relu(correlation.FunctionCorrelation(feat3_0_warped, feat3_1_warped), 0.1, False)
        
        _, _, H, W = feat2_0.shape
        feat = self.up3(torch.cat([feat3_0, feat3_1, volume, feat, offset], dim=1), H, W)
        offset = self.offset2(feat)
        
                        
        feat2_0_warped = self._resample(offset[:,:2, ...], feat2_0)
        feat2_1_warped = self._resample(offset[:,2:, ...], feat2_1)
        volume = F.leaky_relu(correlation.FunctionCorrelation(feat2_0_warped, feat2_1_warped), 0.1, False)
        
        _, _, H, W = feat1_0.shape
        feat = self.up4(torch.cat([feat2_0, feat2_1, volume, feat, offset], dim=1), H, W)
        offset = self.offset1(feat)
        
                        
        feat1_0_warped = self._resample(offset[:,:2, ...], feat1_0)
        feat1_1_warped = self._resample(offset[:,2:, ...], feat1_1)
        volume = F.leaky_relu(correlation.FunctionCorrelation(feat1_0_warped, feat1_1_warped), 0.1, False)
        
        _, _, H, W = feat0_0.shape
        feat = self.up5(torch.cat([feat1_0, feat1_1, volume, feat, offset], dim=1), H, W)
        offset = self.offset0(feat)
        
            
        feat0_0_warped = self._resample(offset[:, :2, ...], feat0_0)
        feat0_1_warped = self._resample(offset[:, 2:, ...], feat0_1)
        volume = F.leaky_relu(correlation.FunctionCorrelation(feat0_0_warped, feat0_1_warped), 0.1, False)
        
        feat = self.up6(torch.cat([feat0_0, feat0_1, volume, feat, offset], dim=1))
        offset = self.offset(feat)
        
        I0_warped = self._resample(offset[:, :2, ...], I0)
        I1_warped = self._resample(offset[:, 2:, ...], I1)
        feat0_0_warped = self._resample(offset[:, :2, ...], feat0_0)
        feat0_1_warped = self._resample(offset[:, 2:, ...], feat0_1)
        
        weights = self.blend(torch.cat([I0_warped, I1_warped, feat0_0_warped, feat0_1_warped], dim=1))
        weights = weights.unsqueeze(2)
        weights = weights.repeat(1, 1, c, 1, 1)
        x = weights[:,0,:, :,:]*I0_warped + weights[:,1,:,:,:]*I1_warped
        
        if self.context:
            x = x + self.enhance(torch.cat([I0_warped, I1_warped, feat0_0_warped, feat0_1_warped], dim=1))
            
        return x, offset, weights, None, None    
    
    def _resample(self, offset, frame, interpolation=False):
        # frame: b*c*h*w
        # offset: b*(2*k)*h*w
        # filters: b*k*h*w
            
        b, c, H, W = frame.shape
                        
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

        gridX = torch.tensor(gridX, requires_grad=False,).cuda()
        gridY = torch.tensor(gridY, requires_grad=False,).cuda()
        
        u = offset[:,0,...]
        v = offset[:,1,...]
        X = gridX.unsqueeze(0).expand_as(u).float() + u
        Y = gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        X = 2.*(X/W - 0.5)
        Y = 2.*(Y/H - 0.5)
        
        x = F.grid_sample(frame, torch.stack((X, Y), dim=-1), align_corners=True,
                                     mode='bilinear', padding_mode='reflection')
        return x
