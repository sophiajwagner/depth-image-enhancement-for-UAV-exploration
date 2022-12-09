import torch.utils.data
import os
import skimage.io as io
from torch.utils.data import Dataset
import numpy as np


class DepthDataset(Dataset):
    def __init__(self, hparams, transform=None):
        super().__init__()
        self.input_path = os.path.join(hparams['input_data_path'])
        self.gt_path = os.path.join(hparams['gt_data_path'])
        self.transform = transform
        # maybe add quality later

    def __len__(self):
        return len(os.listdir(*[self.input_path]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_name = os.path.join(self.input_path, 'image_'+str(idx+1)+'.tif')
        input = io.imread(input_name)
        input = (input - 0.5)/19.5 #normalize image to be in [0,1]
        input[np.isnan(input)] = 0
        input[input == -np.inf] = 0
        input[input == np.inf] = 1
        #print(np.max(input))
        #print(np.min(input))

        gt_name = os.path.join(self.gt_path, 'image_'+str(idx+1)+'.tif')
        gt = io.imread(gt_name)
        gt = (gt - 0.5)/19.5 #normalize image to be in [0,1]
        gt[np.isnan(gt)] = 0
        gt[gt == -np.inf] = 0
        gt[gt == np.inf] = 1
        #print(np.max(gt))
        #print(np.min(gt))

        sample = {'input': input, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample