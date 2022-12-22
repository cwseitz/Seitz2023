import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skimage.measure import label
from skimage.io import imread
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries
from skimage.color import rgb2gray
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class UNetModel(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = 3
        self.padding = 1
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.double_conv.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal(m.weight, mean=0.0, std=np.sqrt(2/(self.in_channels*self.kernel_size**2)))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.kernel_size = 2
        self.in_channels = in_channels
        self.stride = 2
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=self.kernel_size, stride=self.stride)
            torch.nn.init.normal(self.up.weight, mean=0.0, std=np.sqrt(2/(in_channels*self.kernel_size**2)))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.kernel_size = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size)
        torch.nn.init.normal(self.conv.weight, mean=0.0, std=np.sqrt(2/(in_channels*self.kernel_size**2)))
    def forward(self, x):
        return self.conv(x)
        
            

