import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms



class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        #https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.5),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.5),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.encoder(x)



class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)



class StereoAutoencoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        #z1 = z1.view(z1.size(0), -1)
        #z2 = z2.view(z2.size(0), -1)
        z = torch.cat((z1, z2), dim=1)

        return self.decoder(z)

