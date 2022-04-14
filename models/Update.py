#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from turtle import forward
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class UVLoss():
    def __init__():
        pass

    def forward():
        pass

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = UVLoss() #Custom loss function
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, c):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=self.args.momentum) #lr = learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.01)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad() #To avoid gradient accumulation
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels, c)
                loss.backward() 
                optimizer.step()  
                scheduler.step() 
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) #return model and loss value

