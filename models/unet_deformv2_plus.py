import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import correlation
from . import ResBlock
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d#ModulatedDeformConv2dFunction
from torch.nn.modules.utils import _pair, _single
import math

# modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply
    
class DeformConvWarp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, offset_group=1):
        super(DeformConvWarp, self).__init__()
        self.DCN = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size= kernel_size,
            stride= stride,
            padding= padding,
            dilation = dilation,
            deformable_groups = groups
        )
        
    def forward(self, x, offset, mask):
        return self.DCN(x, offset, mask)

    
class DeformConvWarpNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=False):
        super(DeformConvWarpNorm, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.kernel = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         self.kernel**2))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d(x, offset, mask, 
                                       (torch.softmax(self.weight, dim=2)).reshape((self.out_channels, 
                                                                                  self.in_channels // self.groups,
                                                                                  self.kernel, self.kernel)), 
                                       self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

    
class DeformConvWarpNormSingle(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=False):
        super(DeformConvWarpNormSingle, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.kernel = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(1, 1,
                         self.kernel**2))

        self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = 1
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        c = x.shape[1]
        return modulated_deform_conv2d(x, offset, mask, 
                                      self.weight.reshape((1, 1, self.kernel, self.kernel)).repeat(c, 
                                                      1, 1, 1), 
                                      self.bias,
                                      self.stride, self.padding,
                                      self.dilation, c,
                                      self.deform_groups)
    

    
class down(nn.Module):
    expansion = 1

    def __init__(self, inChannels, outChannels, kernel_size):
        super(down, self).__init__()
        self.conv = nn.Conv2d(inChannels,  outChannels, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(outChannels,  outChannels, 3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(outChannels,  outChannels, 3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # weight_init.xavier_normal()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.conv(x))
        
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

    
class up(nn.Module):
    expansion = 1

    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(outChannels,  outChannels, 3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(inChannels,  outChannels, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # weight_init.xavier_normal()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.conv1x1(residual)
        out = self.relu(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_ch=3, kernel=2, groups=1, num_input_frames=2, num_output_frames=1, context=False):
        
        super(UNet, self).__init__()
        self.in_ch = in_ch
        
        self.kernel = kernel
        self.groups = groups
        self.offset_channels = 2*2*kernel*kernel*groups
        self.weight_channels = 2
        self.context = context

        self.ex1 = nn.Sequential(
                    nn.Conv2d(self.in_ch, 16, kernel_size=7, stride=1, padding=3),
                    nn.PReLU(),
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
                    nn.PReLU(),
                    )
        self.ex2 = down(16, 32, 3)
        self.ex3 = down(32, 64, 3)
        self.ex4 = down(64, 96, 3)
        self.ex5 = down(96, 128, 3)
        self.ex6 = down(128, 196, 3)
        
        self.up1   = up(196+196+81, 256)
        self.up2   = up(256+128+128+81+self.offset_channels, 196)
        self.up3   = up(196+96+96+81+self.offset_channels, 128)
        self.up4   = up(128+64+64+81+self.offset_channels, 96)
        self.up5   = up(96+32+32+81+self.offset_channels, 64)
        self.up6   = up(64+16+16+81+self.offset_channels, 64)

        self.motion1 = up(196+196, 81)
        self.motion2 = up(128+128, 81)
        self.motion3 = up(96+96, 81)
        self.motion4 = up(64+64, 81)
        self.motion5 = up(32+32, 81)
        self.motion6 = up(16+16+in_ch*2, 81)
                
        self.offset4 = nn.Conv2d(256, self.offset_channels, 3, stride=1, padding=1)
        self.offset3 = nn.Conv2d(196, self.offset_channels, 3, stride=1, padding=1)
        self.offset2 = nn.Conv2d(128, self.offset_channels, 3, stride=1, padding=1)
        self.offset1 = nn.Conv2d(96, self.offset_channels, 3, stride=1, padding=1)
        self.offset0 = nn.Conv2d(64, self.offset_channels, 3, stride=1, padding=1)
        self.offset  = nn.Conv2d(64, self.offset_channels, 3, stride=1, padding=1)
        
        self.mask4 = nn.Conv2d(256, 2*kernel*kernel*groups, 3, stride=1, padding=1)
        self.mask3 = nn.Conv2d(196, 2*kernel*kernel*groups, 3, stride=1, padding=1)
        self.mask2 = nn.Conv2d(128, 2*kernel*kernel*groups, 3, stride=1, padding=1)
        self.mask1 = nn.Conv2d(96, 2*kernel*kernel*groups, 3, stride=1, padding=1)
        self.mask0 = nn.Conv2d(64, 2*kernel*kernel*groups, 3, stride=1, padding=1)
        self.mask  = nn.Conv2d(64, 2*kernel*kernel*groups, 3, stride=1, padding=1)
        
        self.deform4 = DeformConvWarp(128, 128, self.kernel, padding=self.kernel//2)
        self.deform3 = DeformConvWarp(96, 96, self.kernel, padding=self.kernel//2)
        self.deform2 = DeformConvWarp(64, 64, self.kernel, padding=self.kernel//2)
        self.deform1 = DeformConvWarp(32, 32, self.kernel, padding=self.kernel//2)
        self.deform0_feat = DeformConvWarp(16, 16, self.kernel, padding=self.kernel//2)
        self.deform0 = DeformConvWarp(in_ch, in_ch, self.kernel, padding=self.kernel//2)
        self.deform_feat = DeformConvWarp(16, 16, self.kernel, padding=self.kernel//2)
        self.deform = DeformConvWarpNormSingle()

        self.blend = nn.Sequential(
                nn.Conv2d(in_channels=in_ch + in_ch + 16*2, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
                nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
                nn.Softmax(dim=1))

        self.relu = nn.PReLU()
        if self.context:
            self.enhance = ResBlock.__dict__['MultipleBasicBlock_4'](3 + 3 + 16*2, 64)
    
    def forward(self, input_frames, use_cuda=False, channel_wise_norm=False):
        b, c, t, h, w = input_frames.shape
        I0 = input_frames[:,:, 0, :, :].contiguous()
        I1 = input_frames[:,:, 1, :, :].contiguous()
        if channel_wise_norm:
            mean_R = input_frames[:,0, :, :, :].mean(dim=(1,2,3))
            std_R = input_frames[:,0, :, :, :].std(dim=(1,2,3))
            mean_G = input_frames[:,1, :, :, :].mean(dim=(1,2,3))
            std_G = input_frames[:,1, :, :, :].std(dim=(1,2,3))
            mean_B = input_frames[:,2, :, :, :].mean(dim=(1,2,3))
            std_B = input_frames[:,2, :, :, :].std(dim=(1,2,3))
            I0 = I0.permute((1, 2, 3, 0))
            I1 = I1.permute((1, 2, 3, 0))
            I0[0,...] = (I0[0,...] - mean_R)/(std_R+0.000001)
            I1[0,...] = (I1[0,...] - mean_R)/(std_R+0.000001)
            I0[1,...] = (I0[1,...] - mean_G)/(std_G+0.000001)
            I1[1,...] = (I1[1,...] - mean_G)/(std_G+0.000001)
            I0[2,...] = (I0[2,...] - mean_B)/(std_B+0.000001)
            I1[2,...] = (I1[2,...] - mean_B)/(std_B+0.000001)
            I0 = I0.permute((3, 0, 1, 2))
            I1 = I1.permute((3, 0, 1, 2))
        else:
            I0 = I0/255.
            I1 = I1/255.
         
        h_padded = False
        w_padded = False
        if h % 32 != 0:
            pad_h = 32 - (h % 32)
            I0 = F.pad(I0, (0, 0, 0, pad_h), mode='reflect')
            I1 = F.pad(I1, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w % 32 != 0:
            pad_w = 32 - (w % 32)
            I0 = F.pad(I0, (0, pad_w, 0, 0), mode='reflect')
            I1 = F.pad(I1, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

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

        volume = self.motion1(torch.cat([feat5_0, feat5_1], dim=1))
        feat = self.up1(torch.cat([feat5_0, feat5_1, volume], dim=1))
        offset = self.offset4(feat)
        mask = torch.sigmoid(self.mask4(feat))
        offset_0 = offset[:, :2*self.groups*self.kernel*self.kernel, ...].contiguous()
        offset_1 = offset[:, 2*self.groups*self.kernel*self.kernel:, ...].contiguous()
        mask_0 = mask[:, :self.groups*self.kernel*self.kernel, ...].contiguous()
        mask_1 = mask[:, self.groups*self.kernel*self.kernel:, ...].contiguous()

        _, _, H, W = feat4_0.shape
        offset = self._upsample(offset, H, W)
        mask = self._upsample(mask, H, W)
        feat = self._upsample(feat, H, W)
        offset_0 = offset[:, :2*self.groups*self.kernel*self.kernel, ...].contiguous()
        offset_1 = offset[:, 2*self.groups*self.kernel*self.kernel:, ...].contiguous()
        mask_0 = mask[:, :self.groups*self.kernel*self.kernel, ...].contiguous()
        mask_1 = mask[:, self.groups*self.kernel*self.kernel:, ...].contiguous()
        feat4_0_warped = self.relu(self.deform4(feat4_0.contiguous(), offset_0, mask_0))
        feat4_1_warped = self.relu(self.deform4(feat4_1.contiguous(), offset_1, mask_1))
        volume = self.motion2(torch.cat([feat4_0_warped, feat4_1_warped], dim=1))

        feat = self.up2(torch.cat([feat4_0_warped, feat4_1_warped, volume, feat, offset], dim=1))
        offset = offset + self.offset3(feat)
        mask = torch.sigmoid(self.mask3(feat))

        _, _, H, W = feat3_0.shape 
        offset = self._upsample(offset, H, W)
        mask = self._upsample(mask, H, W)
        feat = self._upsample(feat, H, W)
        offset_0 = offset[:, :2*self.groups*self.kernel*self.kernel, ...].contiguous()
        offset_1 = offset[:, 2*self.groups*self.kernel*self.kernel:, ...].contiguous()
        mask_0 = mask[:, :self.groups*self.kernel*self.kernel, ...].contiguous()
        mask_1 = mask[:, self.groups*self.kernel*self.kernel:, ...].contiguous()
               
        feat3_0_warped = self.deform3(feat3_0, offset_0, mask_0)
        feat3_1_warped = self.deform3(feat3_1, offset_1, mask_1)
        volume = self.motion3(torch.cat([feat3_0_warped, feat3_1_warped], dim=1))
        
        feat = self.up3(torch.cat([feat3_0_warped, feat3_1_warped,  volume, feat, offset], dim=1))
        offset = offset + self.offset2(feat)
        mask = torch.sigmoid(self.mask2(feat))

        _, _, H, W = feat2_0.shape
        offset = self._upsample(offset, H, W)
        mask = self._upsample(mask, H, W)
        feat = self._upsample(feat, H, W)
        offset_0 = offset[:, :2*self.groups*self.kernel*self.kernel, ...].contiguous()
        offset_1 = offset[:, 2*self.groups*self.kernel*self.kernel:, ...].contiguous()
        mask_0 = mask[:, :self.groups*self.kernel*self.kernel, ...].contiguous()
        mask_1 = mask[:, self.groups*self.kernel*self.kernel:, ...].contiguous()
               
        feat2_0_warped = self.deform2(feat2_0, offset_0, mask_0)
        feat2_1_warped = self.deform2(feat2_1, offset_1, mask_1)
        volume = self.motion4(torch.cat([feat2_0_warped, feat2_1_warped], dim=1))

        feat = self.up4(torch.cat([feat2_0_warped, feat2_1_warped, volume, feat, offset], dim=1))
        offset = offset + self.offset1(feat)
        mask = torch.sigmoid(self.mask1(feat))

        _, _, H, W = feat1_0.shape
        offset = self._upsample(offset, H, W)
        mask = self._upsample(mask, H, W)
        feat = self._upsample(feat, H, W)
        offset_0 = offset[:, :2*self.groups*self.kernel*self.kernel, ...].contiguous()
        offset_1 = offset[:, 2*self.groups*self.kernel*self.kernel:, ...].contiguous()
        mask_0 = mask[:, :self.groups*self.kernel*self.kernel, ...].contiguous()
        mask_1 = mask[:, self.groups*self.kernel*self.kernel:, ...].contiguous()
               
        feat1_0_warped = self.deform1(feat1_0, offset_0, mask_0)
        feat1_1_warped = self.deform1(feat1_1, offset_1, mask_1)
        volume = self.motion5(torch.cat([feat1_0_warped, feat1_1_warped], dim=1))

        feat = self.up5(torch.cat([feat1_0_warped, feat1_1_warped, volume, feat, offset], dim=1))
        offset = offset + self.offset0(feat)
        mask = torch.sigmoid(self.mask0(feat))

        _, _, H, W = feat0_0.shape
        offset = self._upsample(offset, H, W)
        mask = self._upsample(mask, H, W)
        feat = self._upsample(feat, H, W)
        offset_0 = offset[:, :2*self.groups*self.kernel*self.kernel, ...].contiguous()
        offset_1 = offset[:, 2*self.groups*self.kernel*self.kernel:, ...].contiguous()
        mask_0 = mask[:, :self.groups*self.kernel*self.kernel, ...].contiguous()
        mask_1 = mask[:, self.groups*self.kernel*self.kernel:, ...].contiguous()
          
        feat0_0_warped = self.deform0_feat(feat0_0, offset_0, mask_0)
        feat0_1_warped = self.deform0_feat(feat0_1, offset_1, mask_1)
        I0_warped = self.deform0(I0, offset_0, mask_0)
        I1_warped = self.deform0(I1, offset_1, mask_1)
        volume = self.motion6(torch.cat([feat0_0_warped, I0_warped, feat0_1_warped, I1_warped], dim=1))

        feat = self.up6(torch.cat([feat0_0_warped, feat0_1_warped, volume, feat, offset], dim=1))
        offset = offset + self.offset(feat)
        mask = torch.sigmoid(self.mask(feat))

        offset_0 = offset[:, :2*self.groups*self.kernel*self.kernel, ...].contiguous()
        offset_1 = offset[:, 2*self.groups*self.kernel*self.kernel:, ...].contiguous()
        mask_0 = mask[:, :self.groups*self.kernel*self.kernel, ...].contiguous()
        mask_1 = mask[:, self.groups*self.kernel*self.kernel:, ...].contiguous()
        
        I0_warped = self.deform(I0, offset_0, mask_0)
        I1_warped = self.deform(I1, offset_1, mask_1)
        feat0_0_warped = self.deform_feat(feat0_0, offset_0, mask_0)
        feat0_1_warped = self.deform_feat(feat0_1, offset_1, mask_1)
        
        weights = self.blend(torch.cat([I0_warped, I1_warped, feat0_0_warped, feat0_1_warped], dim=1))
        weights = weights.unsqueeze(2)
        weights = weights.repeat(1, 1, 3, 1, 1)
        x = weights[:,0,:, :,:]*I0_warped[:,:3, :,:] + weights[:,1,:,:,:]*I1_warped[:,:3, :,:]
        
        x_inter = None
        res = None
        if self.context:
            res = self.enhance(torch.cat([I0_warped, I1_warped, feat0_0_warped, feat0_1_warped], dim=1)) 
            x = x + res
        
        if h_padded:
            x = x[:, :, 0:h, :]
        if w_padded:
            x = x[:, :, :, 0:w]
         
        if channel_wise_norm:
            x = x.permute((1, 2, 3, 0))
            x[0,...] = x[0,...]*(std_R+0.000001) + mean_R
            x[1,...] = x[1,...]*(std_G+0.000001) + mean_G
            x[2,...] = x[2,...]*(std_B+0.000001) + mean_B
            x = x.permute((3, 0, 1, 2))
            
        return x, offset, weights, res, x_inter

    def _upsample(self, x, H, W):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x