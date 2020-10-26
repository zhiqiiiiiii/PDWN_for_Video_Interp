import os
from PIL import Image
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from time import time
from util.util import rescale
from PIL import Image
import torch.nn.functional as F
import random
from skimage import io
import skimage
#from mypath import Path

class VideoDataset(Dataset):

    def __init__(self, dataset='middle', split='val', num_input_frames=2, step=8, interpolation=False, resize=False, crop=False, flip=False, reverse=False):
        # modify the path
        if split == 'val':
            if num_input_frames == 4:
                self.output_dir = './eval-data-4'
            else:
                self.output_dir = './eval-data'
        elif split == 'other-4':
            self.output_dir = './other-data-4/'
        else:
            if num_input_frames == 4:
                self.output_dir = './other-data-4/'
            else:
                self.output_dir = './other-data'       
           
        self.split = split
        self.videos = os.listdir(self.output_dir)
        self.flip = flip
        self.crop = crop
        self.reverse = reverse
        self.num_input_frames = num_input_frames

        self.crop_size = 352

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')
        


    def __len__(self):
        return len(self.videos)
      
    
    def __getitem__(self, index):
        video = self.videos[index]
        print(os.path.join(self.output_dir, video, 'frame10.png'))
        
        buffer = []
        if self.num_input_frames == 2:
            frame = io.imread(os.path.join(self.output_dir, video, 'frame10.png'))
            if len(frame.shape) == 2:
                frame = skimage.color.gray2rgb(frame)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
#             frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
            frame = frame/255.0
            buffer.append(frame)
            
            if not self.split == 'val':
                frame = io.imread(os.path.join(self.output_dir, video, 'frame10i11.png'))
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
#                 frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
                frame = frame/255.0
                buffer.append(frame)

            frame = io.imread(os.path.join(self.output_dir, video, 'frame11.png'))
            if len(frame.shape) == 2:
                frame = skimage.color.gray2rgb(frame)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
#             frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
            frame = frame/255.0
            buffer.append(frame)
        else:
            frame = io.imread(os.path.join(self.output_dir, video, 'frame09.png'))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
#             frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
            frame = frame/255.0
            buffer.append(frame)
            
            frame = io.imread(os.path.join(self.output_dir, video, 'frame10.png'))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
#             frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
            frame = frame/255.0
            buffer.append(frame)
        
            if not self.split == 'val':
                frame = io.imread(os.path.join(self.output_dir, video, 'frame10i11.png'))
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
#                 frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
                frame = frame/255.0
                buffer.append(frame)

            frame = io.imread(os.path.join(self.output_dir, video, 'frame11.png'))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
#             frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
            frame = frame/255.0
            buffer.append(frame)
            
            frame = io.imread(os.path.join(self.output_dir, video, 'frame12.png'))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
#             frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
            frame = frame/255.0
            buffer.append(frame)
        buffer = np.stack(buffer, axis=0)
        buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))
        return buffer
    
    def check_integrity(self):
        if not os.path.exists(self.output_dir):
            return False
        else:
            return True

    def random_crop(self, buffer, output_size):
        h, w, c = buffer[0].shape
        th, tw = output_size
        if w == tw and h == th:
            i = 0 
            j = 0
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        
        for k, frame in enumerate(buffer):
            buffer[k] = frame[i:i+th, j:j+tw, :]
        return buffer
    
        
    def random_flip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer
  

    def random_reverse(self, buffer):
        
        if np.random.random() < 0.3:
            buffer.reverse()

        return buffer