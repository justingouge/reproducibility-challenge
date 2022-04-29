#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from cProfile import label
from cmath import sqrt, tau
from tkinter import Variable
from turtle import pos, towards
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import random, math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from bch_master.bch import generate, encode
from utils.sampling import celebA_split, celebA_get_dict_users, voxCeleb, mnistUV
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNMnistUV, CNNCelebA
from models.Fed import FedAvg

#This function is used for accuracy calculations
def accuracy(net_glob, dataset, user_vectors):
    net_glob.eval()
    tau_vals = []
    #For accuracy calc, we will just check which class the input is closer to (accept or reject)
    for i in range(len(user_vectors)):
        tau_vals.append(0.51)

    loss, accuracy = calc_TPR(net_glob, dataset, user_vectors, tau_vals)
    return loss, accuracy

#This is the warm-up phase from the main FedUV algorithm which calculates tau values for each individual user 
def warm_up(net_glob, dataset, user_vectors, q):
    net_glob.eval()
    tau_vals = []

    with torch.no_grad():
        for user in dataset:
            e_vector = []
            loaded_data = DataLoader(user)
            for pic_num, (images, labels) in enumerate(loaded_data):
                images, labels = images.to(args.device), labels.to(args.device)
                output = net_glob(images)
                curr_user_vector = user_vectors[str(labels[0].item())]
                user_vector = np.asarray(curr_user_vector, dtype=np.float32)
                user_vector = torch.tensor(user_vector, dtype=torch.float32).to(args.device)
                dot_result = torch.dot(user_vector, output[0])
                multiply_result = torch.multiply(torch.tensor(1/args.code_length).to(args.device), dot_result)
                e_vector.append(multiply_result.cpu().detach().item())

            i = math.floor(len(dataset[0]) * (1 - q))
            e_vector.sort()
            tau_vals.append(e_vector[i])

    assert(len(tau_vals) == len(dataset))
    return tau_vals

#Calculates TPR specifically.... we could just the q value, but this will give us an exact value!
def calc_TPR(net_glob, dataset, user_vectors, tau_vals):
    net_glob.eval()

    total_accepted = 0
    curr_user = 0
    accum_loss = 0

    with torch.no_grad():
        for user in dataset:
            loaded_data = DataLoader(user)
            for i, (images, labels) in enumerate(loaded_data):
                images, labels = images.to(args.device), labels.to(args.device)
                output = net_glob(images)
                curr_user_vector = user_vectors[str(labels[0].item())]
                user_vector = np.asarray(curr_user_vector, dtype=np.float32)
                user_vector = torch.tensor(user_vector, dtype=torch.float32).to(args.device)                
                dot_result = torch.dot(user_vector, output[0])
                multiply_result = torch.multiply(torch.tensor(1/args.code_length).to(args.device), dot_result)
                e = multiply_result
                accum_loss += max(0, 1 - e)
                if e >= tau_vals[curr_user]:
                    total_accepted+=1
            
            curr_user+=1

    assert(curr_user==len(dataset))
    return accum_loss/(len(dataset) * len(dataset[0])), total_accepted/(len(dataset) * len(dataset[0]))  

#Calculates the FPR for each user, using a bad vector strategy (should be rejects!)
def calc_FPR(net_glob, dataset, user_vectors, tau_vals):
    net_glob.eval()

    total_accepted = 0
    curr_user = 0

    with torch.no_grad():
        for user in dataset:
            loaded_data = DataLoader(user)
            for i, (images, labels) in enumerate(loaded_data):
                images, labels = images.to(args.device), labels.to(args.device)
                output = net_glob(images)                
                
                #Try a bad user_vector and see if it is accepted!
                bad_vector_num = random.sample(user_vectors.keys(), 1)

                #We need a DIFFERENT vector than the user we are checking
                while np.array_equal(user_vectors[bad_vector_num[0]], user_vectors[str(labels[0].item())]):
                    bad_vector_num = random.sample(user_vectors.keys(), 1)

                bad_vector = user_vectors[str(bad_vector_num[0])]
                  
                user_vector = np.asarray(bad_vector, dtype=np.float32)
                user_vector = torch.tensor(user_vector, dtype=torch.float32).to(args.device)
                dot_result = torch.dot(user_vector, output[0])
                multiply_result = torch.multiply(torch.tensor(1/args.code_length).to(args.device), dot_result)
                e = multiply_result
                if e >= tau_vals[curr_user]:
                    total_accepted += 1

            curr_user+=1

    assert(curr_user==len(dataset))
    return total_accepted/(len(dataset) * len(dataset[0]))  

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # load dataset and split users
    if args.dataset == 'mnistUV':
        pass
    elif args.dataset == 'celebA':
        trans_celebA = transforms.Compose([transforms.ToTensor(), transforms.Resize([64, 64]), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CelebA('../data/celebA/', split='all', download=False, transform=trans_celebA, target_type="identity")
        dataset_train, dataset_validation, dataset_test, dataset_new_users = celebA_split(dataset)

        dict_users = celebA_get_dict_users(dataset_train, args.num_users)
        #Save the stuff for debugging
        torch.save(dataset_train, "dataset_train")
        torch.save(dataset_validation, "dataset_validation")
        torch.save(dataset_test, "dataset_test")
        torch.save(dataset_new_users, "dataset_new_users")
        torch.save(dict_users, "dict_users")
    elif args.dataset == 'voxCeleb':
        pass
    else:
        exit('Error: unrecognized dataset')

    #Load the stuff for debugging
    dataset_train = torch.load("dataset_train")
    dataset_validation = torch.load("dataset_validation")
    dataset_test = torch.load("dataset_test")
    dataset_new_users = torch.load("dataset_new_users")
    dict_users = celebA_get_dict_users(dataset_train, args.num_users)
    torch.save(dict_users, "dict_users")
    dict_users = torch.load("dict_users")

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #THIS SECTION is for generating the secret user vectors!
    #We generate secret vectors for the users that take part in training and also the new users we use in the verification part.
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #Generate codewords for UV
    user_vectors = {}
    generate(args.code_length, 0, args.d_min, "bch_master/file") #Generate polynomial for BCH

    #Making secret vectors for our identities for model training
    for user_id in dict_users.keys():
        b_u = [int(x) for x in bin(int(user_id))[2:]]
        r_u = [ random.randint(0, 1) for i in range(args.message_length - len(b_u)) ]
        m_u = np.concatenate((b_u, r_u))
        assert(len(m_u) == args.message_length)
        v_u = encode("bch_master/file.npz", m_u, block=False)
        
        #Need to make user vector exact length
        if len(v_u) < args.code_length:
            missing_amount = args.code_length - len(v_u)
            v_u = np.concatenate((v_u, np.zeros(missing_amount)))
        elif len(v_u) > args.code_length:
            v_u = v_u[0:args.code_length]

        assert(len(v_u) == args.code_length)
        user_vectors[user_id] = v_u

    torch.save(user_vectors, "user_vectors_127")
    user_vectors = torch.load("user_vectors_127")

    #Making secret vectors for the new_users 
    new_user_vectors = {}

    for user_id in celebA_get_dict_users(dataset_new_users, args.num_users).keys():
        b_u = [int(x) for x in bin(int(user_id))[2:]]
        r_u = [ random.randint(0, 1) for i in range(args.message_length - len(b_u)) ]
        m_u = np.concatenate((b_u, r_u))
        assert(len(m_u) == args.message_length)
        v_u = encode("bch_master/file.npz", m_u, block=False)
        
        #Need to make user vector exact length
        if len(v_u) < args.code_length:
            missing_amount = args.code_length - len(v_u)
            v_u = np.concatenate((v_u, np.zeros(missing_amount)))
        elif len(v_u) > args.code_length:
            v_u = v_u[0:args.code_length]

        assert(len(v_u) == args.code_length)
        new_user_vectors[user_id] = v_u

    torch.save(new_user_vectors, "new_user_vectors_127")
    new_user_vectors = torch.load("new_user_vectors_127")

    #Make user vectors {-1, 1}^c
    for vector in user_vectors.values():
        for i in range(len(vector)):
            if vector[i] == 0:
                vector[i] = -1
                
    for vector in new_user_vectors.values():
        for i in range(len(vector)):
            if vector[i] == 0:
                vector[i] = -1

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #THIS SECTION TRAINING WITH POSITIVE LOSS FUNCTION!
    #To keep the privacy with each user, the authors created a training strategy using only positive loss!
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    pos_loss_model = CNNCelebA(args=args).to(args.device)

    #print(net_glob)
    pos_loss_model.train()

    # copy weights
    w_glob = pos_loss_model.state_dict()

    # training
    loss_train_pos = []
    loss_test_pos = []
    acc_train_pos_loss, acc_test_pos_loss = [], []

    if args.all_clients: 
        
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.array(random.sample(dict_users.keys(), m)) #randomly choose users
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train[dict_users[idx]], idxs=dict_users[idx], count=len(dataset_train[0])) 
            w, loss = local.train(net=copy.deepcopy(pos_loss_model).to(args.device), user_vectors=user_vectors, complete=False) 
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w)) #appending different w to w_local
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals) #this is our actual aggregated w

        # copy weight to net_glob
        pos_loss_model.load_state_dict(w_glob)

        # get loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train_pos.append(loss_avg)

        #Build accuracy list
        if iter % (math.floor(args.epochs/80)) == 0:
            _, train_acc = accuracy(pos_loss_model, dataset_train, user_vectors)
            test_loss, test_acc = accuracy(pos_loss_model, dataset_validation, user_vectors)
            acc_train_pos_loss.append(train_acc)
            acc_test_pos_loss.append(test_acc)
            loss_test_pos.append(test_loss)
            pos_loss_model.train()

        #print(loss_train_pos[iter])
        
    
    torch.save(pos_loss_model.state_dict(), "pos_loss_state_dict_127")
    torch.save(acc_train_pos_loss, "acc_train_pos_loss_127")
    torch.save(acc_test_pos_loss, "acc_test_pos_loss_127")
    torch.save(loss_train_pos, "loss_train_pos_127")
    torch.save(loss_test_pos, "loss_test_pos_127")

    pos_loss_model = CNNCelebA(args=args).to(args.device)
    pos_loss_model.load_state_dict(torch.load("pos_loss_state_dict_127"))
    acc_train_pos_loss = torch.load("acc_train_pos_loss_127")
    acc_test_pos_loss = torch.load("acc_test_pos_loss_127")
    loss_train_pos = torch.load("loss_train_pos_127")
    loss_test_pos = torch.load("loss_test_pos_127")
   
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #THIS SECTION TRAINING WITH COMPLETE LOSS FUNCTION! Unfortunately, I don't collect results from this part because it will take many days to run :( instead, I hope to compare my positive loss results to
    #what we see in the paper. The below code works, it will just take too long to run
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    

    complete_loss_model = CNNCelebA(args=args).to(args.device)
    complete_loss_model.train()

    # copy weights
    w_glob = complete_loss_model.state_dict()

    # training
    loss_train_complete = []
    loss_test_complete = []
    acc_train_complete_loss, acc_test_complete_loss = [], []

    if args.all_clients: 
        #print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.array(random.sample(dict_users.keys(), m)) #randomly choose user
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train[dict_users[idx]], idxs=dict_users[idx], count=len(dataset_train[0]))
            w, loss = local.train(net=copy.deepcopy(complete_loss_model).to(args.device), user_vectors=user_vectors, complete=True) 
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w)) #appending different w to w_local
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals) #this is our actual aggregated w

        # copy weight to net_glob
        complete_loss_model.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        #print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_complete.append(loss_avg)

        #Get round accuracies
        if iter % (math.floor(args.epochs/80)) == 0:            
            _, train_acc = accuracy(complete_loss_model, dataset_train, user_vectors)
            test_loss, test_acc = accuracy(complete_loss_model, dataset_validation, user_vectors)
            acc_train_complete_loss.append(train_acc)
            acc_test_complete_loss.append(test_acc)
            loss_test_complete.append(test_loss)
            complete_loss_model.train()


    torch.save(complete_loss_model.state_dict(), "complete_loss_state_dict")
    torch.save(acc_train_complete_loss, "acc_train_complete_loss")
    torch.save(acc_test_complete_loss, "acc_test_complete_loss")
    torch.save(loss_train_complete, "loss_train_complete")
    torch.save(loss_test_complete, "loss_test_complete")

    complete_loss_model = CNNCelebA(args=args).to(args.device)
    complete_loss_model.load_state_dict(torch.load("pos_loss_state_dict"))
    acc_train_complete_loss = torch.load("acc_train_complete_loss")
    acc_test_complete_loss = torch.load("acc_test_complete_loss")
    loss_train_complete = torch.load("loss_train_complete")
    loss_test_complete = torch.load("loss_test_complete")

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #THIS SECTION is the verification part of the FedUV algorithm.
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    q_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tpr_vals_train = []
    fpr_vals_train = []
    tpr_vals_test = []
    fpr_vals_test = []
    tpr_vals_unknown = []
    fpr_vals_unknown = []

    for q in q_vals:
            
        #Verifiction with train 
        tau_values = warm_up(pos_loss_model, dataset_train, user_vectors, q)
        _, tpr = calc_TPR(pos_loss_model, dataset_train, user_vectors, tau_values)
        fpr = calc_FPR(pos_loss_model, dataset_train, user_vectors, tau_values)
        tpr_vals_train.append(tpr)
        fpr_vals_train.append(fpr)

        #Verification with test set known users
        tau_values = warm_up(pos_loss_model, dataset_test, user_vectors, q)
        _, tpr = calc_TPR(pos_loss_model, dataset_test, user_vectors, tau_values)
        fpr = calc_FPR(pos_loss_model, dataset_test, user_vectors, tau_values)
        tpr_vals_test.append(tpr)
        fpr_vals_test.append(fpr)

        #Verification with test set unknown users
        tau_values = warm_up(pos_loss_model, dataset_new_users, new_user_vectors, q)
        _, tpr = calc_TPR(pos_loss_model, dataset_new_users, new_user_vectors, tau_values)
        fpr = calc_FPR(pos_loss_model, dataset_new_users, new_user_vectors, tau_values)
        tpr_vals_unknown.append(tpr)
        fpr_vals_unknown.append(fpr)

    torch.save(tpr_vals_train, "tpr_vals_train_127")
    torch.save(fpr_vals_train, "fpr_vals_train_127")
    torch.save(tpr_vals_test, "tpr_vals_test_127")
    torch.save(fpr_vals_test, "fpr_vals_test_127")
    torch.save(tpr_vals_unknown, "tpr_vals_unknown_127")
    torch.save(fpr_vals_unknown, "fpr_vals_unkown_127")

    tpr_vals_train = torch.load("tpr_vals_train_127")
    fpr_vals_train = torch.load("fpr_vals_train_127")
    tpr_vals_test = torch.load("tpr_vals_test_127")
    fpr_vals_test = torch.load("fpr_vals_test_127")
    tpr_vals_unknown = torch.load("tpr_vals_unknown_127")
    fpr_vals_unknown = torch.load("fpr_vals_unkown_127")
    
    #Creating tables/graphs for results
    epoch_arr = []
    for i in range(80):
        epoch_arr.append(i * math.floor(args.epochs/80))

    for val in range(len(loss_test_pos)):
        loss_test_pos[val] = loss_test_pos[val].item()

    plt.figure()
    plt.plot(range(20000), loss_train_pos, label='Training Loss')
    plt.plot(epoch_arr, loss_test_pos, label='Test Loss')
    plt.legend()
    plt.title('Training and Test Loss with CelebA')
    plt.ylabel('Loss')
    plt.xlabel('Round Number')
    plt.savefig('./save/Training and Test Loss with CelebA(127).png')

    plt.figure()
    plt.plot(epoch_arr, acc_train_pos_loss, 'g', label='Train (w/o L_neg)')
    plt.plot(epoch_arr, acc_test_pos_loss, 'r--', label='Test (w/o L_neg)')
    plt.legend()
    plt.title('Training and Test Accuracy with CelebA')
    plt.ylabel('Accuracy')
    plt.xlabel('Round Number')
    plt.savefig('./save/Training and Test Accuracy with CelebA(127).png')

    #Borrowing values from original paper
    softmax_vals_tpr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    softmax_vals_fpr = [0, 0, 0, 0, 0, 0, 0, 0.004, 0.01, 0.02]
    fedaws_vals_tpr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fedaws_vals_fpr = [0, 0, 0, 0, 0, 0.00125, 0.005, 0.015, 0.028, 0.04]

    plt.figure()
    plt.plot(softmax_vals_fpr, softmax_vals_tpr, label='softmax')
    plt.plot(fedaws_vals_fpr, fedaws_vals_tpr, label='FedAws')
    plt.plot(fpr_vals_train, tpr_vals_train, label='FedUV(127)')
    plt.legend()
    plt.title('Training dataset ROC Curve with FedUV(127)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./save/Training dataset ROC Curve with FedUV(127).png')

    #Borrowing values from original paper
    softmax_vals_tpr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    softmax_vals_fpr = [0, 0, 0, 0, 0, 0, 0.00125, 0.0025, 0.01, 0.03]
    fedaws_vals_tpr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fedaws_vals_fpr = [0, 0, 0, 0.005, 0.0075, 0.01, 0.0125, 0.018, 0.0325, 0.075]

    plt.figure()
    plt.plot(softmax_vals_fpr, softmax_vals_tpr, label='softmax')
    plt.plot(fedaws_vals_fpr, fedaws_vals_tpr, label='FedAws')
    plt.plot(fpr_vals_test, tpr_vals_test, label='FedUV(127)')
    plt.legend()
    plt.title('Test dataset ROC Curve with FedUV(127)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./save/Test dataset ROC Curve with FedUV(127).png')

    #Borrowing values from original paper
    softmax_vals_tpr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    softmax_vals_fpr = [0, 0, 0, 0.0001, 0.001, 0.002, 0.025, 0.1, 0.25, 0.41]
    fedaws_vals_tpr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fedaws_vals_fpr = [0, 0.0025, 0.0035, 0.005, 0.0125, 0.0175, 0.0325, 0.049, 0.0675, 0.0875]

    plt.figure()
    plt.plot(softmax_vals_fpr, softmax_vals_tpr, label='softmax')
    plt.plot(fedaws_vals_fpr, fedaws_vals_tpr, label='FedAws')
    plt.plot(fpr_vals_unknown, q_vals, label='FedUV(127)')
    plt.legend()
    plt.title('Unknown Users ROC Curve with FedUV(127)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./save/Unknown Users ROC Curve with FedUV(127).png')

