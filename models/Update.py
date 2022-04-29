#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from collections import UserString
from turtle import forward
from venv import create
from xml.sax.handler import DTDHandler
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
from sklearn import metrics

#This class is just the object for training
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, count=0):
        self.args = args
        self.selected_clients = []
        self.count = count
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, user_vectors, complete):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01) #lr = learning rate
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
           
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad() #To avoid gradient accumulation
    
                embeddings = net(images)        
                curr_user_vector = user_vectors[str(labels[0].item())]
                if complete == True:
                    loss = uv_loss_complete(self.args, embeddings, curr_user_vector, user_vectors)
                else:
                    loss = uv_loss_pos(self.args, embeddings, curr_user_vector)
                x = loss.grad_fn
                loss.backward()
                optimizer.step()  
                scheduler.step() 
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) #return model and loss value

#UV Loss Functions
#Positive loss function from the paper
def uv_loss_pos(args, nn_output, user_vector):
 #   assert(len(nn_output[0]) == len(user_vector))
    accum_loss = torch.tensor(0).to(args.device)
    user_vector = np.asarray(user_vector, dtype=np.float32)
    user_vector = torch.tensor(user_vector, dtype=torch.float32).to(args.device)

    #Calculate batch loss
    for embedding in nn_output:     
        dot_result = torch.dot(user_vector, embedding)
        multiply_result = torch.multiply(torch.tensor(1/args.code_length).to(args.device), dot_result)
        subtract_result = torch.subtract(torch.tensor(1).to(args.device), multiply_result)
        max_result = torch.max(torch.tensor(0).to(args.device), subtract_result)
        accum_loss = torch.add(accum_loss, max_result) 

    return torch.divide(accum_loss, torch.tensor(len(nn_output)).to(args.device))

#Complete loss function from the paper
def uv_loss_complete(args, nn_output, user_vector, all_vectors):
    assert(len(nn_output[0]) == len(user_vector))
    accum_loss = torch.tensor(0).to(args.device)
    user_vector = np.asarray(user_vector, dtype=np.float32)
    user_vector = torch.tensor(user_vector, dtype=torch.float32).to(args.device)

    #Calculate batch loss
    for embedding in nn_output:
        dot_result = torch.dot(user_vector, embedding)
        multiply_result = torch.multiply(torch.tensor(1/args.code_length).to(args.device), dot_result)
        subtract_result = torch.subtract(torch.tensor(1).to(args.device), multiply_result)
        l_pos = torch.max(torch.tensor(0).to(args.device), subtract_result)
        
        loss_set = torch.tensor([0]).to(args.device)

        for vector in all_vectors.keys():
            if not np.array_equal(user_vector, all_vectors[vector]):
                user_vector = np.asarray(all_vectors[vector], dtype=np.float32)
                user_vector = torch.tensor(user_vector, dtype=torch.float32).to(args.device)
                dot_result = torch.dot(user_vector, embedding)
                multiply_result = torch.multiply(torch.tensor(1/args.code_length).to(args.device), dot_result)
                subtract_result = torch.subtract(torch.tensor(1).to(args.device), multiply_result)
                l_neg = torch.max(torch.tensor(0).to(args.device), subtract_result)
                loss_set = torch.cat((loss_set, l_neg.reshape(1)), 0)

        l_neg = torch.max(loss_set)
        curr_loss = torch.add(l_pos, l_neg)
        accum_loss = torch.add(accum_loss, curr_loss)

    #return torch.tensor(float(accum_loss / len(embeddings)), requires_grad=True)
    return torch.divide(accum_loss, torch.tensor(len(nn_output)).to(args.device))