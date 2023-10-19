# implementation by Rohan
import os
import torch
import glob
from PIL import Image
import random
import clip
from tqdm.notebook import tqdm
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from torch.distributions.gamma import Gamma
device = "cuda:2"
a_u = 1
b_u = 0

a_minus = 10
b_minus = 0

a_plus = 5
b_plus = 0

def sample_w(U, s_matrix):
    BS = s_matrix.shape[0]
    s_plus = s_matrix.masked_select(torch.eye(BS).bool().to(device))
    s_minus = s_matrix.masked_select(~torch.eye(BS).bool().to(device)).\
            reshape(BS, -1)
    w_plus_dist = Gamma(torch.tensor(1+a_plus).float().to(device),\
            U*s_plus + b_plus)
    w_minus_dist = Gamma(torch.tensor(a_minus).float().to(device),\
            U.unsqueeze(1)*s_minus + b_minus)
    
#     print("s_minus:", s_minus.size())
#     print("w_minus_dist:", (U.unsqueeze(1)*s_minus).size())
    w_plus = w_plus_dist.sample()
#     print("w_plus:",w_plus)
    w_minus = w_minus_dist.sample()
#     print("w_minus:", w_minus)
    tmp = torch.cat((w_plus[:-1].unsqueeze(1),\
            w_minus.transpose(1, 0)), dim=1)
    w_matrix = torch.cat((tmp.view(-1),\
            w_plus[-1].unsqueeze(0))).view(BS, BS)
    print("w_matrix:",w_matrix)
    return w_matrix

def sample_u(w_matrix, sim_matrix):
    full_mat = w_matrix * sim_matrix
    rate_param = b_u + full_mat.sum(dim=1)
    u_dist = Gamma(torch.tensor(a_u).float().to(device),\
            rate_param.float())
    # print(rate_param)
    # print(u_dist.sample())
    return u_dist.sample()

BATCH_SIZE =5
seed = 42
torch.manual_seed(seed)
weights = torch.ones(BATCH_SIZE, BATCH_SIZE).to(device)
similarity_score = torch.rand(BATCH_SIZE, BATCH_SIZE).to(device)
for _ in range(1):
    U = sample_u(weights, similarity_score)
    # print("U:", U.size())
    weights = sample_w(U, similarity_score)
# print(weights)