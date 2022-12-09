import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms



class RGBEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        #https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)
        


class DepthEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        #https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)



class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.decoder = nn.Sequential(
            nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)



class StereoAutoencoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.stereo_encoder = RGBEncoder(hparams)
        self.depth_encoder = DepthEncoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, x_left, x_right, x_depth):
        z_left = self.stereo_encoder(x_left)
        z_right = self.stereo_encoder(x_right)
        z_depth = self.depth_encoder(x_depth)
        
        z = torch.cat((z_left, z_depth, z_right), dim=1)
        return self.decoder(z)

