import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy import special as sp
from scipy.constants import pi

import torch
from tensorboardX import SummaryWriter

from helper_function.utils import args_parser
from helper_function.utils import LocalUpdate, test_inference
from helper_function.utils import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from helper_function.utils import get_dataset, average_weights, exp_details
from AFLweight import asy_average_weights, asy_average_weights_weight

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

vehicle_model = []
for i in range(args.num_users):
    vehicle_model.append(copy.deepcopy(glmodel))


# copy weights
glweights = glmodel.state_dict()

# Training
trloss, tracc = [], []
tr_step_loss = []
tr_step_acc = []
vlacc, net_ = [], []
cvloss, cvacc = [], []
print_epoch = 2
vllossp, cnt = 0, 0


for ep in tqdm(range(args.epochs)):
    locweights, locloss = [], []
    print(f'\n | Global Training Round : {ep+1} |\n')

    glmodel.train()

    user_id = range(args.num_users)


    beta_s = [0] * args.num_users
    for i in range(args.num_users):
        beta_s[i] = (i + 1) / 10

    delta = [0] * args.num_users
    for i in range(args.num_users):
        delta[i] = 1e+9 * 1.5 * (0.1*i + 1.5)


    CPUcycles = 1e+5
    kexi = 0.9
    localtime = [0] * args.num_users
    beta_lt = [0] * args.num_users
    for i in range(args.num_users):
        localtime[i] = (48000 * (i + 1) / args.num_users) * CPUcycles / delta[i] -1
        beta_lt[i] = kexi ** localtime[i]



    def complexGaussian(row=1, col=1, amp=1.0):
        real = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
        img = np.random.normal(size=[row, col])[0] * np.sqrt(0.5)
        return amp * (real + 1j * img)
    aaa = complexGaussian(1, 1)[0]
    H1 =[aaa] * args.num_users


    alpha = 2
    path_loss = [0] * args.num_users
    dis = [0] * args.num_users
    for i in range(args.num_users):
        dis[i] = -500 + 30 * i
        path_loss[i] = 1 / np.power(np.linalg.norm(dis[i]), alpha)


    Hight_RSU = 10
    width_lane = 5
    velocity = 20
    lamda = 7
    x_0 = np.array([1, 0, 0])
    P_B = np.array([0, 0, Hight_RSU])
    P_m = [0] * args.num_users
    rho = [0] * args.num_users
    for i in range(args.num_users):
        P_m[i] = np.array([dis[i], width_lane, Hight_RSU])
        rho[i] = sp.j0(2 * pi * velocity * np.dot(x_0, (P_B - P_m[i])) / (np.linalg.norm(P_B - P_m[i]) * lamda))


    for i in range(args.num_users):


        H1[i] = rho[i] * H1[i] + complexGaussian(1, 1, np.sqrt(1 - rho[i] * rho[i]))[0]
        ddd = abs(H1[i])


    transpower = 250

    sigma2 = 1e-9
    bandwidth = 1e+3
    tr = [0] * args.num_users
    sinr = [0] * args.num_users
    for i in range(args.num_users):
        sinr[i] = transpower * abs(H1[i]) * path_loss[i] / sigma2
        tr[i] = np.log2(1 + sinr[i]) * bandwidth


    w_size = 5000
    c_c = [0] * args.num_users
    for i in range(args.num_users):
        c_c[i] = w_size / tr[i] -1

    beta_ct = [0] * args.num_users
    epuxilong = 0.9
    for i in range(args.num_users):
        beta_ct[i] = epuxilong ** c_c[i]


    for j in user_id:
        vehicle_start_time = time.time()
        local_net = copy.deepcopy(vehicle_model[j])
        local_net.to(device)
        locmdl = LocalUpdate(args=args, dataset=trdata, idxs=usgrp[j], logger=logger)
        w, loss, localmodel = locmdl.asyupdate_weights(model=copy.deepcopy(glmodel), global_round=ep)

        locweights.append(copy.deepcopy(w))
        locloss.append(copy.deepcopy(loss))


        glmodel, glweights = asy_average_weights_weight(vehicle_idx=j, global_model=glmodel, local_model=localmodel,
                                                        gamma=args.gamma, local_param1=1,
                                                        local_param2=beta_lt[j], local_param3=beta_ct[j])

        globalmodelw,globalmodelloss,globalmodelmodel = locmdl.asyupdate_weights(model=copy.deepcopy(glmodel), global_round=ep)
        print("globalmodelloss is ", globalmodelloss)
        vehicle_model[j] = copy.deepcopy(glmodel)

        step_acc, step_loss = [], []
        glmodel.eval()
        for q in range(args.num_users):
            locmdl = LocalUpdate(args=args, dataset=trdata,
                                 idxs=usgrp[q], logger=logger)
            acc1, loss1 = locmdl.inference(model=glmodel)
            step_acc.append(acc1)
            step_loss.append(loss1)
        tr_step_acc.append(sum(step_acc) / len(step_acc))
        # tr_step_loss.append(sum(step_loss) / len(step_loss))

        # print step training loss after every 'i' rounds
        # if (ep+1) % print_epoch == 0:
        print(f' \nAvg Training Stats after {ep + 1} global rounds:')
        # print(f'step Training Loss : {np.mean(np.array(tr_step_loss))}')
        # print("tr_step_loss is", tr_step_loss)
        print('step Train Accuracy: {:.2f}% \n'.format(100 * tr_step_acc[-1]))

        cost_time = time.time() - vehicle_start_time
        for i in range(args.num_users):  # i从 0 - (args.num_users-1)
            dis[i] += cost_time * velocity
    # print(' tr_step_loss is :', tr_step_loss)


    # avg_ep_loss = sum(locloss) / len(locloss)
    # trloss.append(avg_ep_loss)

    # Calculate avg training accuracy over all users at every epoch
    # epacc, eploss = [], []
    # glmodel.eval()
    # for q in range(args.num_users):
    #     locmdl = LocalUpdate(args=args, dataset=trdata,
    #                               idxs=usgrp[q], logger=logger)
    #     acc, loss = locmdl.inference(model=glmodel)
    #     epacc.append(acc)
    #     eploss.append(loss)
    # tracc.append(sum(epacc)/len(epacc))

    # print global training loss after every 'i' rounds
    # if (ep+1) % print_epoch == 0:
    # print(f'\nAvg Training Stats after {ep+1} epoch rounds:')
    # print(f'ep Training Loss : {np.mean(np.array(trloss))}')
    # print('ep Train Accuracy: {:.2f}% \n'.format(100*tracc[-1]))

# Test inference after completion of training
tsacc, tsloss = test_inference(args, glmodel, tsdata)

print(f' \n Results after {args.epochs} epoch rounds of training:')
# print("|---- Avg Train Accuracy: {:.2f}%".format(100*tracc[-1]))
print("|---- Test Accuracy: {:.2f}%".format(100*tsacc))

# Saving the objects train_loss and train_accuracy:
fname = 'results/models/epo_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)

with open(fname, 'wb') as f:
    pickle.dump([trloss, tracc], f)


# Saving the objects train_loss and train_accuracy:
fname = 'results/models/step_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)

with open(fname, 'wb') as f:
    pickle.dump([tr_step_loss, tr_step_acc], f)

print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

# PLOTTING (optional)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Plot Loss curve
plt.figure()
plt.title('epoch Training Loss')
# plt.title('Training Loss using AFederated Learning')
plt.plot(range(len(trloss)), trloss, color='b')
plt.ylabel('Training loss')
plt.xlabel('Num of epoch Rounds')
plt.savefig('results/AFLweight_epo_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
             format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))




# # Plot Average Accuracy vs Communication rounds
plt.figure()
plt.title('Average Accuracy')
# plt.title('Average Accuracy using Federated Learning')
plt.plot(range(len(tracc)), tracc, color='g')
plt.ylabel('Average Accuracy')
plt.xlabel('Num of Rounds')
plt.savefig('results/AFLweight_epo_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
            format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))


plt.figure()
plt.title('step Training Loss')
# plt.title('Training Loss using AFederated Learning')
plt.plot(range(len(tr_step_loss)), tr_step_loss, color='b')
plt.ylabel('Training loss')
plt.xlabel('Num of step Rounds')
plt.savefig('results/AFLweight_step_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
             format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))


plt.figure()
plt.title('step Average Accuracy')
# plt.title('Average Accuracy using Federated Learning')
plt.plot(range(len(tr_step_acc)), tr_step_acc, color='g')
plt.ylabel('Average Accuracy')
plt.xlabel('Num of step Rounds')
plt.savefig('results/AFLweight_step_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
            format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))


plt.figure()
plt.title('step Average Accuracy/loss')
# plt.title('Average Accuracy using Federated Learning')

plt.ylabel('Average Accuracy/loss')
plt.xlabel('Num of step Rounds')

plt.plot(tr_step_acc, color='blue', linewidth=1.5, linestyle='-', label='tr_step_acc')
plt.plot(tr_step_loss, color='green', linewidth=1.5, linestyle='-', label='tr_step_loss')
plt.savefig('results/AFLweight_step_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc and loss.png'.
            format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs))

