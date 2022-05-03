import torch
import os
import numpy as np
from tfrecord.torch.dataset import TFRecordDataset


tf_dir = "/Users/eash/UIUC/CS 598- Vision/project/tf_sstables/"
num_files = None
# prefix = "train"
prefix = "val"

p3d = {}
audio = {}

file_idx = 0

for filename in os.listdir(tf_dir):
    if num_files and file_idx>=num_files:
        break
    f = os.path.join(tf_dir, filename)
    if os.path.isfile(f) and prefix in filename:
        print("Processing %s" % filename)
    else:
        # print("Skipping %s" % filename)
        continue
    file_idx += 1
    dataset = TFRecordDataset(f, index_path=None, description=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, data in enumerate(loader):
        key = data['motion_name'][0].numpy()
        key = ''.join([chr(x) for x in key])
        print("Vidoe name:", key)
        pn, pd = data['motion_sequence_shape'][0]
        pn=pn.item()
        pd=pd.item()
        p3d[key] = data['motion_sequence'].view((pn, pd))
        an, ad = data['audio_sequence_shape'][0]
        an=an.item()
        ad=ad.item()
        audio[key] = data['audio_sequence'].view((an, ad))


torch.save(p3d, "data/p3d_"+prefix+".pth")
torch.save(audio, "data/audio_"+prefix+".pth")