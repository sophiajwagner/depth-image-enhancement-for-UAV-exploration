import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms



class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        self.hparams = hparams
        #https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
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
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.encoder(x)



class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()

        self.hparams = hparams
        self.decoder = nn.Sequential(
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



class Autoencoder(nn.Module):
    def __init__(self, hparams):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

