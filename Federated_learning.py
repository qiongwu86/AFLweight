import os



import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from helper_function.utils import args_parser
from helper_function.utils import LocalUpdate, test_inference
from helper_function.utils import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from helper_function.utils import get_dataset, average_weights, exp_details

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
start_time = time.time()

# define paths
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')

args = args_parser()
exp_details(args)

if args.gpu == 1:
    torch.cuda.set_device(0)
device = 'cuda' if args.gpu else 'cpu'

# load dataset and user groups
trdata, tsdata, usgrp = get_dataset(args)

# BUILD MODEL
if args.model == 'cnn':
    if args.dataset == 'mnist':
        glmodel = CNNMnist(args=args)
    elif args.dataset == 'fmnist':
        glmodel = CNNFashion_Mnist(args=args)
    elif args.dataset == 'cifar':
        glmodel = CNNCifar(args=args)
elif args.model == 'mlp':
    imsize = trdata[0][0].shape
    input_len = 1
    for x in imsize:
        input_len *= x
        glmodel = MLP(dim_in=input_len, dim_hidden=64,
                           dim_out=args.num_classes)
else:
    exit('Error: unrecognized model')

glmodel.to(device)
glmodel.train()
print(glmodel)

# copy weights
glweights = glmodel.state_dict()

# Training
trloss, tracc = [], []
vlacc, net_ = [], []
cvloss, cvacc = [], []
print_epoch = 2
vllossp, cnt = 0, 0

for ep in tqdm(range(args.epochs)):
    locweights, locloss = [], []
    print(f'\n | Global Training Round : {ep+1} |\n')

    glmodel.train()
    m = max(int(args.frac * args.num_users), 1)
    user_id = np.random.choice(range(args.num_users), m, replace=False)

    for j in user_id:
        locmdl = LocalUpdate(args=args, dataset=trdata,
                                  idxs=usgrp[j], logger=logger)
        w, loss = locmdl.update_weights(
            model=copy.deepcopy(glmodel), global_round=ep)
        locweights.append(copy.deepcopy(w))
        locloss.append(copy.deepcopy(loss))

    # update global weights
    glweights = average_weights(locweights)

    # update global weights
    glmodel.load_state_dict(glweights)

    avg_loss = sum(locloss) / len(locloss)
    trloss.append(avg_loss)

    # Calculate avg training accuracy over all users at every epoch
    epacc, eploss = [], []
    glmodel.eval()
    for q in range(args.num_users):
        locmdl = LocalUpdate(args=args, dataset=trdata,
                                  idxs=usgrp[q], logger=logger)
        acc, loss = locmdl.inference(model=glmodel)
        epacc.append(acc)
        eploss.append(loss)
    tracc.append(sum(epacc)/len(epacc))

    # # print global training loss after every 'i' rounds
    # if (ep+1) % print_epoch == 0:
    #     print(f' \nAvg Training Stats after {ep+1} global rounds:')
    #     print(f'Training Loss : {np.mean(np.array(trloss))}')
    #     print('Train Accuracy: {:.2f}% \n'.format(100*tracc[-1]))

    # print global training loss after every 'i' rounds

    print(f' \nAvg Training Stats after {ep+1} global rounds:')
    print(f'Training Loss : {np.mean(np.array(trloss))}')
    print('Train Accuracy: {:.2f}% \n'.format(100*tracc[-1]))

    print('Training Loss', np.mean(np.array(trloss)))
# Test inference after completion of training
tsacc, tsloss = test_inference(args, glmodel, tsdata)

print(f' \n Results after {args.epochs} global rounds of training:')
print("|---- Avg Train Accuracy: {:.2f}%".format(100*tracc[-1]))
print("|---- Test Accuracy: {:.2f}%".format(100*tsacc))
print("testloss is :", tsloss)
# Saving the objects train_loss and train_accuracy:
fname = 'results/models/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)

with open(fname, 'wb') as f:
    pickle.dump([trloss, tracc], f)

print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

# PLOTTING (optional)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Plot Loss curve
plt.figure()
plt.title('Training Loss')
# plt.title('Training Loss using Federated Learning')
plt.plot(range(len(trloss)), trloss, color='b')
plt.ylabel('Training loss')
plt.xlabel('Num of Rounds')
plt.savefig('results/FL_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
             format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))
#
# # Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('Average Accuracy')
# plt.title('Average Accuracy using Federated Learning')
plt.plot(range(len(tracc)), tracc, color='g')
plt.ylabel('Average Accuracy')
plt.xlabel('Num of Rounds')
plt.savefig('results/FL_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
            format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))
