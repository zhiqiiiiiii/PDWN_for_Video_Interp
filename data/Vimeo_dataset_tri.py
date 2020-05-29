import os
from PIL import Image
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from time import time
from PIL import Image
import torch.nn.functional as F
import random

class VideoDataset(Dataset):

    def __init__(self, dataset='vimeo', split='train', clip_len=3, 
                     interpolation=False, crop=True, flip=True, reverse=True, negative=False):
        self.output_dir = '/beegfs/rw1691/data/VIMEO-7/vimeo_septuplet/sequences/'
        if split == 'train':
            list_file = '/beegfs/rw1691/data/VIMEO-7/vimeo_septuplet/tri_trainlist.txt'
        else:
            list_file = '/beegfs/rw1691/data/VIMEO-7/vimeo_septuplet/sequences/tri_testlist.txt'
        
        self.videos = []
        with open(list_file, "r") as f:
            for video in f.readlines():
                self.videos.append(video.strip() )
        print("finish sequences read ")
        self.videos = self.videos[:-1]
        self.clip_len = clip_len
        self.interpolation = interpolation
        self.flip = flip
        self.crop = crop
        self.reverse = reverse
        self.negative = negative

        # The following three parameters are chosen as described in the paper section 4.1
        self.crop_size = 352

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        print('Number of {} clips: {:d}, Number of {} total frames'.format(split, len(self.videos), 
                                                                           len(self.videos)*3))


    def __len__(self):
        return len(self.videos)
    
    
    def __getitem__(self, index):
        video = self.videos[index]
        # Loading and preprocessing.

        buffer = []
        frame = cv2.imread(os.path.join(self.output_dir, video, 'im1.png'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = normalize(frame, negative=self.negative)
        #frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
        buffer.append(frame)
        
        frame = cv2.imread(os.path.join(self.output_dir, video, 'im3.png'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = normalize(frame, negative=self.negative)
        #frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
        buffer.append(frame)
        
        frame = cv2.imread(os.path.join(self.output_dir, video, 'im4.png'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = normalize(frame, negative=self.negative)
        #frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
        buffer.append(frame)
        
        frame = cv2.imread(os.path.join(self.output_dir, video, 'im5.png'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = normalize(frame, negative=self.negative)
        #frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
        buffer.append(frame)

        frame = cv2.imread(os.path.join(self.output_dir, video, 'im7.png'))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)
        frame = normalize(frame, negative=self.negative)
        #frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
        buffer.append(frame)
        
        if self.crop:
            buffer = self.random_crop(buffer, (256, 256))
        if self.flip:
            buffer = self.random_flip(buffer)
        if self.reverse:
            buffer = self.random_reverse(buffer)
        buffer = np.stack(buffer, axis=0)
        buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))
        return buffer

    def normalize(self, frame, negative=True):
        if negative:
            return 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
        else:
            return frame/255.0
    
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

# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
#     train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)

#     for i, sample in enumerate(train_loader):
#         inputs = sample[0]
#         print(inputs.size())

#         if i == 1:
#             break





