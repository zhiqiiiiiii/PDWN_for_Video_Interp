import os
from PIL import Image
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from time import time
from skimage import io


class VideoDataset(Dataset):

    def __init__(self, dataset='ucf101', clip_len=6, num_input_frames=4, num_output_frames=2, interpolation=False, beyondMSE=False, DVF=False):
        # modify the path
        if beyondMSE:
            self.root_dir = '.../MathieuICLR16TestCode/UCF101frm10p'
        if DVF:
            if interpolation:
                self.root_dir = '.../ucf101_interp_ours'
            else:
                self.root_dir = '.../ucf101_extrap_ours'
        self.videos = os.listdir(self.root_dir)
        
        self.clip_len = 6
        self.num_input_frames = num_input_frames
        self.num_output_frames = num_output_frames
        self.interpolation = interpolation
        
        
        print('Number of {} clips: {:d}'.format("test", len(self.videos)))

    def __len__(self):
        return len(self.videos) 
    
    def __getitem__(self, index):
        video = self.videos[index]
        if self.num_input_frames == 4:
            buffer = []
            for i in range(1, self.num_input_frames+1):
                frame = io.imread(os.path.join(self.root_dir, video, 'pred_'+str(i)+'.png'))
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
#                 frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
                frame = frame/255.0
                buffer.append(frame)

            flow = []
            flow_map = io.imread(os.path.join(self.root_dir, video, 'pred_4_flow.png'))
#             flow_map = cv2.cvtColor(flow_map, cv2.COLOR_BGR2RGB)
            flow_map = flow_map.astype(np.float32)
            flow_map = flow_map/255.0
            flow.append(flow_map)

            for i in range(1, self.num_output_frames+1):
                frame = io.imread(os.path.join(self.root_dir, video, 'target_'+str(i)+'.png'))
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
#                 frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
                frame = frame/255.0
                buffer.append(frame)

                flow_map = io.imread(os.path.join(self.root_dir, video, 'target_'+str(i)+'_flow.png'))
#                 flow_map = cv2.cvtColor(flow_map, cv2.COLOR_BGR2RGB)
                flow_map = flow_map.astype(np.float32)
                flow_map = flow_map/255.0
                flow.append(flow_map)

            buffer = np.stack(buffer, axis=0)
            buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))

            flow = np.stack(flow, axis=0)
            flow = torch.from_numpy(flow.transpose((3, 0, 1, 2)))

            return buffer, flow
        else:
            buffer = []
            if self.interpolation:
                l = [0,2]
            else:
                l = [0,1]
            for i in l:
                frame = io.imread(os.path.join(self.root_dir, video, 'frame_0'+str(i)+'.png'))
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
#                 frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
                frame = frame/255.0
                buffer.append(frame)
            if self.interpolation:
                frame = io.imread(os.path.join(self.root_dir, video, 'frame_01_gt.png'))
            else:
                frame = io.imread(os.path.join(self.root_dir, video, 'frame_02_gt.png'))
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
#             frame = 2.0*(frame - frame.min()) / (frame.max() - frame.min()) - 1.0
            frame = frame/255.0
            buffer.append(frame)
            buffer = np.stack(buffer, axis=0)
            buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))
            return buffer

