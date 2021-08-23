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
from skimage import io

#from mypath import Path

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, data_root, dataset='vimeo', split='train', clip_len=3, 
                     crop=False, flip=True, reverse=True, negative=False, eraser=False, normalize=True, self_ensemble=False):
        self.data_root = data_root
        print("loading dataset from ", self.data_root)
        self.output_dir = os.path.join(data_root, 'sequences')
        if split == 'train':
            list_file = os.path.join(self.data_root, 'tri_trainlist.txt')
            self.is_train = True
        else:
            list_file = os.path.join(self.data_root, 'tri_testlist.txt')
            self.is_train = False
        
        self.videos = []
        with open(list_file, "r") as f:
            for video in f.readlines():
                self.videos.append(video.strip() )
        print("finish sequences read ")
        self.videos = self.videos[:-1]
        self.clip_len = clip_len
        self.flip = flip
        self.crop = crop
        self.reverse = reverse
        self.negative = negative
        self.eraser = eraser
        self.normalize = normalize
        self.self_ensemble = self_ensemble

        # The following three parameters are chosen as described in the paper section 4.1
        self.crop_size = 256

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
        frame1 = io.imread(os.path.join(self.output_dir, video, 'im1.png'))
        frame1 = frame1.astype(np.float32)
        if self.normalize:
            frame1 = self.normalize(frame1, negative=self.negative)

        frame2 = io.imread(os.path.join(self.output_dir, video, 'im2.png'))
        frame2 = frame2.astype(np.float32)
        if self.normalize:
            frame2 = self.normalize(frame2, negative=self.negative)
        
        frame3 = io.imread(os.path.join(self.output_dir, video, 'im3.png'))
        frame3 = frame3.astype(np.float32)
        if self.normalize:
            frame3 = self.normalize(frame3, negative=self.negative)
        
        if self.is_train and self.eraser:
            frame1, frame3 = self.eraser_transform(frame1, frame3)
        
        buffer.append(frame1)
        buffer.append(frame2)
        buffer.append(frame3)
        
        if self.crop:
            buffer = self.random_crop(buffer, (256, 256))
        if self.flip:
            buffer = self.random_flip(buffer)
        if self.reverse:
            buffer = self.random_reverse(buffer)
            
        if self.self_ensemble:
            buffer_flip = []
            for i, frame in enumerate(buffer):
                buffer_flip.append(cv2.flip(frame, flipCode=1))
            buffer_mirror = []
            for i, frame in enumerate(buffer):
                buffer_mirror.append(cv2.flip(frame, flipCode=0))
            buffer_reverse = buffer
            buffer_reverse.reverse()
            buffer_flip_reverse = []
            for i, frame in enumerate(buffer):
                buffer_flip_reverse.append(cv2.flip(frame, flipCode=1))
            buffer_flip_reverse.reverse()
            buffer_rotate_90 = []
            for i, frame in enumerate(buffer):
                buffer_rotate_90.append(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
            buffer_rotate_180 = []
            for i, frame in enumerate(buffer):
                buffer_rotate_180.append(cv2.rotate(frame, cv2.ROTATE_180))
            buffer_rotate_90_inverse = []
            for i, frame in enumerate(buffer):
                buffer_rotate_90_inverse.append(cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
            buffer_flip = np.stack(buffer_flip, axis=0)
            buffer_flip = torch.from_numpy(buffer_flip.transpose((3, 0, 1, 2)))
            buffer_mirror = np.stack(buffer_mirror, axis=0)
            buffer_mirror = torch.from_numpy(buffer_mirror.transpose((3, 0, 1, 2)))
            buffer_reverse = np.stack(buffer_reverse, axis=0)
            buffer_reverse = torch.from_numpy(buffer_reverse.transpose((3, 0, 1, 2)))
            buffer_flip_reverse = np.stack(buffer_flip_reverse, axis=0)
            buffer_flip_reverse = torch.from_numpy(buffer_flip_reverse.transpose((3, 0, 1, 2)))
            buffer_rotate_90 = np.stack(buffer_rotate_90, axis=0)
            buffer_rotate_90 = torch.from_numpy(buffer_rotate_90.transpose((3, 0, 1, 2)))
            buffer_rotate_180 = np.stack(buffer_rotate_180, axis=0)
            buffer_rotate_180 = torch.from_numpy(buffer_rotate_180.transpose((3, 0, 1, 2)))
            buffer_rotate_90_inverse = np.stack(buffer_rotate_90_inverse, axis=0)
            buffer_rotate_90_inverse = torch.from_numpy(buffer_rotate_90_inverse.transpose((3, 0, 1, 2)))
            
        buffer = np.stack(buffer, axis=0)
        buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))
        if self.self_ensemble:
            return [buffer, buffer_flip, buffer_mirror, buffer_reverse, 
                    buffer_flip_reverse, buffer_rotate_90, buffer_rotate_180, buffer_rotate_90_inverse]
        return buffer
    
    def normalize(self, frame, negative=False):
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
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=0)
        return buffer
  
    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < 0.8:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def random_reverse(self, buffer):
        
        if np.random.random() < 0.5:
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