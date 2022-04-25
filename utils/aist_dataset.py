from tracemalloc import start
from torch.utils.data import Dataset, DataLoader
import numpy as np
from h5py import File
import scipy.io as sio
# from utils import data_utils
from matplotlib import pyplot as plt
import torch
from tfrecord.torch.dataset import TFRecordDataset

import os 

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion3d.py
'''


class Datasets(Dataset):

    def __init__(self, data_dir, input_n=20, output_n=40, skip_rate=40, sample_rate=2, prefix="train", num_files=None):
        self.path_to_data = data_dir
        self.in_n = input_n*sample_rate
        self.out_n = output_n*sample_rate
        self.sample_rate = sample_rate
        self.p3d = {}
        self.audio = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n

        file_idx=0
        key=0
        for filename in os.listdir(self.path_to_data):
            if num_files and file_idx>=num_files:
                break
            f = os.path.join(self.path_to_data, filename)
            if os.path.isfile(f) and prefix in f:
                print("Processing %s" % filename)
            else:
                # print("Skipping %s" % filename)
                continue
            file_idx += 1
            dataset = TFRecordDataset(f, index_path=None, description=None)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1)

            for i, data in enumerate(loader):
                pn, pd = data['motion_sequence_shape'][0]
                pn=pn.item()
                pd=pd.item()
                self.p3d[key] = data['motion_sequence'].view((pn, pd))
                an, ad = data['audio_sequence_shape'][0]
                an=an.item()
                ad=ad.item()
                self.audio[key] = data['audio_sequence'].view((an, ad))

                start_frames = np.arange(0, pn-seq_len+1, skip_rate)
                self.data_idx = self.data_idx + [(key, sf) for sf in start_frames]

                assert (start_frames<self.p3d[key].shape[0]).all()
                # print(key, start_frames, self.p3d[key].shape)

                key=key+1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]

        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n, self.sample_rate)
        # ps = np.arange(start_frame + self.in_n, start_frame + self.in_n + self.out_n, self.sample_rate)

        # print(fs[0], ps[-1], self.p3d[key].shape, key, start_frame)

        # return self.p3d[key][fs], self.audio[key][fs], self.p3d[key][ps], self.audio[key][ps]
        return self.p3d[key][fs], self.audio[key][fs]

if __name__ == '__main__':
    print("Testing AIST++ dataset")
    ds = Datasets("/Users/eash/UIUC/CS 598- Vision/project/tf_sstables/")

    dataloader = DataLoader(ds, num_workers=1, batch_size=7, shuffle=True)
    example_batch = next(iter(dataloader))
    print(example_batch, example_batch[0].shape)