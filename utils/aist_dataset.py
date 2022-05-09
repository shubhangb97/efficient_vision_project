from tracemalloc import start
from torch.utils.data import Dataset, DataLoader
import numpy as np
from h5py import File
import scipy.io as sio
# from utils import data_utils
from matplotlib import pyplot as plt
import torch

import os 

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion3d.py
'''


class Datasets(Dataset):

    def __init__(self, p3d_pth, audio_pth, input_n=20, output_n=40, skip_rate=40, sample_rate=2, num_files=None):
        self.in_n = input_n*sample_rate
        self.out_n = output_n*sample_rate
        self.sample_rate = sample_rate
        self.p3d = torch.load(p3d_pth)
        self.audio = torch.load(audio_pth)
        self.data_idx = []
        seq_len = self.in_n + self.out_n

        file_idx = 0
        for key in self.p3d.keys():
            if num_files and file_idx >= num_files:
                break
            file_idx += 1
            p3d_data = self.p3d[key]
            nfs = p3d_data.shape[0]

            start_frames = np.arange(0, nfs-seq_len+1, skip_rate)
            self.data_idx = self.data_idx + [(key, sf) for sf in start_frames]

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]

        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n, self.sample_rate)

        return torch.cat((torch.zeros(fs.shape[0], 6), self.p3d[key][fs]), axis=1), self.audio[key][fs]

if __name__ == '__main__':
    print("Testing AIST++ dataset")
    ds = Datasets(
        p3d_pth="/Users/eash/UIUC/CS 598- Vision/project/efficient_vision_project/data/p3d_train.pth",
        audio_pth="/Users/eash/UIUC/CS 598- Vision/project/efficient_vision_project/data/audio_train.pth",
        num_files=1)

    dataloader = DataLoader(ds, num_workers=1, batch_size=7, shuffle=True)
    example_batch = next(iter(dataloader))
    print(example_batch, example_batch[0].shape)