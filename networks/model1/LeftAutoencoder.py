import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

# CNN model that takes a low-light 3 channel stereo image as input and outputs the corresponding high-light image

class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)



class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)



class Autoencoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

