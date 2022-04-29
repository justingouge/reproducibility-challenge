#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from multiprocessing.context import assert_spawning
import numpy as np
from torchvision import datasets, transforms
import random
import torch

def celebA_split(dataset):
    dataset_train = []
    dataset_validation = []
    dataset_test = []
    dataset_new_users = []
  
    thirty_or_more_images = []
    temp_collection = {}
    
    #Go through the dataset and find people with more than 30 images (warning takes a while)
    for i in range(len(dataset)):
        curr_identity = dataset[i][1].item()
        if str(curr_identity) not in temp_collection:
            temp_collection[str(curr_identity)] = []
     
        if len(temp_collection[str(curr_identity)]) < 30:
            temp_collection[str(curr_identity)].append(dataset[i])
    
    torch.save(temp_collection, "temp_collection")
    temp_collection = torch.load("temp_collection")

    for identity in temp_collection:
        if len(temp_collection[identity]) >= 30:
            thirty_or_more_images.append(temp_collection[identity])

    assert(len(thirty_or_more_images) >= 1000)
    dataset_combined = random.sample(thirty_or_more_images, 1000)

    #Split up organized dataset into 3 parts (training, validation, and test)
    for x in dataset_combined:
        dataset_train.append(x[0:20])
        dataset_validation.append(x[20:25])
        dataset_test.append(x[25:30])

    #Sample images from new users (1 per person) and create our unknown users dataset
    sample_list = temp_collection.keys()
    for user in dataset_train:
        sample_list = list(set(sample_list) - set([str(user[0][1].item())]))
    new_user_ids = random.sample(sample_list, 1000)

    for user in new_user_ids:
        dataset_new_users.append(random.sample(temp_collection[user], 1))

    return dataset_train, dataset_validation, dataset_test, dataset_new_users

#This is just to assign a client in the federated learning setting to an identity from the dataset
def celebA_get_dict_users(dataset, num_users):
    #     """
    #     Sample non-I.I.D client data from celebA dataset
    #     """
    
    dict_users = {}
    for i in range(0, num_users):
        dict_users[str(dataset[i][0][1].item())] = i

    return dict_users

def voxCeleb(dataset, num_users):
    pass

def mnistUV(dataset, num_users):
    pass

