import torch.utils.data
import os
import skimage.io as io
from torch.utils.data import Dataset
import numpy as np

# Dataset that loads a low light depth, left and right image as well as the corresponding high-quality depth image

class StereoDataset(Dataset):
    def __init__(self, hparams, transform=None):
        super().__init__()
        self.data_path = hparams['data_path']
        self.left_input_path = os.path.join(self.data_path,'low_light/low_left_tif')
        self.right_input_path = os.path.join(self.data_path, 'low_light/low_right_tif')
        self.depth_input_path = os.path.join(self.data_path, 'low_light/low_depth_tif')
        self.gt_path = os.path.join(self.data_path, 'high_light/high_depth_tif')
        self.transform = transform

    def __len__(self):
        return len(os.listdir(*[self.left_input_path]))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        left_input_name = os.path.join(self.left_input_path, 'image_'+str(idx+1)+'.tif')
        left_input = io.imread(left_input_name)
        #left_input = left_input[:,:,:3] 
        left_input = np.divide(left_input, 255) #normalize image to be in [0,1]

        right_input_name = os.path.join(self.right_input_path, 'image_'+str(idx+1)+'.tif')
        right_input = io.imread(right_input_name)
        #right_input = right_input[:,:,:3]
        right_input = np.divide(right_input, 255)  # normalize image to be in [0,1]
        
        depth_input_name = os.path.join(self.depth_input_path, 'image_'+str(idx+1)+'.tif')
        depth_input = io.imread(depth_input_name)
        depth_input = (depth_input - 0.5)/19.5 #normalize image to be in [0,1]
        depth_input[np.isnan(depth_input)] = 0
        depth_input[depth_input == -np.inf] = 0
        depth_input[depth_input == np.inf] = 1

        gt_name = os.path.join(self.gt_path, 'image_'+str(idx+1)+'.tif')
        gt = io.imread(gt_name)
        gt = (gt - 0.5)/19.5 #normalize image to be in [0,1]
        gt[np.isnan(gt)] = 0
        gt[gt == -np.inf] = 0
        gt[gt == np.inf] = 1

        sample = {'left_input': left_input, 'right_input': right_input, 'depth_input': depth_input, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample