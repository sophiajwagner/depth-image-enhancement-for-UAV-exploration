import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import os
import skimage.io as io
from plyfile import PlyData


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, transform=None):
        super(DepthDataset, self).__init__()
        self.data_path = hparams['data_path']
        self.input_path = os.path.join(self.data_path,'low_light/low_depth')
        self.gt_path = os.path.join(self.data_path, 'high_light/high_depth')
        self.transform = transform
        # maybe add quality later

    def __len__(self):
        return len(os.listdir(*[self.input_path]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_name = os.path.join(self.input_path, 'image_'+str(idx)+'.png')
        input = io.imread(input_name)
        input /= 255 #normalize image to be in [0,1]

        gt_name = os.path.join(self.gt_path, 'image_'+str(idx)+'.png')
        gt = io.imread(gt_name)
        gt /= 255 #normalize image to be in [0,1]

        sample = {'input': input, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample