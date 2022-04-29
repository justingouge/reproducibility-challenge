#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from cmath import sqrt
import torch
from torch import nn
import torch.nn.functional as F
import math

#CelebA network described by the authors in the paper
class CNNCelebA(nn.Module):
    def __init__(self, args):
        super(CNNCelebA, self).__init__()
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=2, num_channels=256)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.gn4 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.gn5 = nn.GroupNorm(num_groups=2, num_channels=1024)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()
        self.c = args.code_length
        self.fc = nn.Linear(1024, args.code_length)
        self.scale = nn.LayerNorm(self.c) #This IS the scaling function they mention 

    def forward(self, x):
        x = self.gn1(self.pool1(F.relu(self.conv1(x))))
        x = self.gn2(self.pool1(F.relu(self.conv2(x))))
        x = self.gn3(self.pool1(F.relu(self.conv3(x))))
        x = self.gn4(self.pool1(F.relu(self.conv4(x))))
        x = self.gn5(self.pool2(F.relu(self.conv5(x))))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.scale(x)
        
        return x

class CNNVoxCeleb:
    pass

class CNNMnistUV:
    pass

