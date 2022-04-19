#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random

# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) #Sampling num_items training sample from all index, and constructed as a set
#         all_idxs = list(set(all_idxs) - dict_users[i]) #Excluding the previous sample
#     return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300 #200 different "pieces" with 300 image each
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy() #numpy() allows for easier manipulation of data

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels)) 
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     # Takes 2 random piece and combine
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users


# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def celebA_split(dataset):
    dataset_train = []
    dataset_validation = []
    dataset_test = []
    #print(dataset[0][0].shape)
    thirty_or_more_images = []
    temp_collection = {}
    #np.random.shuffle(dataset) #comment out if results are being really weird
    for i in range(200000):
        curr_identity = dataset[i][1].item()
        if str(curr_identity) not in temp_collection:
            temp_collection[str(curr_identity)] = []
        
        if len(temp_collection[str(curr_identity)]) < 30:
            temp_collection[str(curr_identity)].append(dataset[i])
        
    for identity in temp_collection:
        if len(temp_collection[identity]) >= 30:
            thirty_or_more_images.append(temp_collection[identity])

    dataset_combined = random.sample(thirty_or_more_images, 1000)

    for x in dataset_combined:
        dataset_train.append(x[0:20])
        dataset_validation.append(x[20:25])
        dataset_test.append(x[25:30])

    return dataset_train, dataset_validation, dataset_test

def celebA_get_dict_users(dataset, num_users):
    #     """
    #     Sample non-I.I.D client data from celebA dataset
    #     """
    
    dict_users = {}
    for i in range(0, num_users):
        dict_users[dataset[i][1].numpy()] = dataset[i][1].numpy()

    return dict_users

def voxCeleb(dataset, num_users):
    pass

def mnistUV(dataset, num_users):
    pass

# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)

