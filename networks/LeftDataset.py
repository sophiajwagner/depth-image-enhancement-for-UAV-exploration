import torch.utils.data
import os
import skimage.io as io
from torch.utils.data import Dataset
import numpy as np

# Dataset that loads low-light stereo images and the corresponding high-light images

class LeftDataset(Dataset):
    def __init__(self, hparams, transform=None):
        super().__init__()
        self.input_path = os.path.join(hparams['input_data_path'])
        self.gt_path = os.path.join(hparams['gt_data_path'])
        self.transform = transform

    def __len__(self):
        return len(os.listdir(*[self.input_path]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_name = os.path.join(self.input_path, 'image_'+str(idx+1)+'.tif')
        input = io.imread(input_name)
        input = input[:,:,:3] # 4th channel consists of 1s only
        input = np.divide(input, 255) #normalize image to be in [0,1]

        gt_name = os.path.join(self.gt_path, 'image_'+str(idx+1)+'.tif')
        gt = io.imread(gt_name)
        gt = gt[:,:,:3] # 4th channel consists of 1s only
        gt = np.divide(gt, 255) #normalize image to be in [0,1]

        sample = {'input': input, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample