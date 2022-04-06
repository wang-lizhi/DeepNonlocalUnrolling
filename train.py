import math
import numpy as np
import torch
import scipy.io as sio
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torchvision
from utils import *
from utils_reid import *
import os
from torch.utils.data import DataLoader
from datasets import HavardDataset
from model import DNU


def get_dataloader(batch_size=64):
    data_path = 'Training_Data_ICVL_48.mat'
    mask_path = 'Cu_48.mat'
    #If you want to train on the CAVE dataset and test on the KAIST dataset,
    #please add the code in the following annotation:
    #key = 'train_list.txt'
    #file_path = data_path + key
    #file_list = loadpath(file_path)
    #HSI = prepare_data(data_path, file_list, 26)
    hd = HavardDataset(data_path, mask_path)
    print(len(hd))
    train_dataloader = DataLoader(hd, batch_size=batch_size, shuffle=True)
    return train_dataloader, len(hd)

def convert_2_tensor(patch, mask, TVT):
    [bs, r, c, nC] = patch.shape
    patch_Phi = mask
    patch_Phi = patch_Phi  #/ 31.0
    patch_x_tensor = patch.view(bs, r, c, nC).permute(0, 3, 1, 2)
    patch_Phi_tensor = patch_Phi.view(bs, r, c, nC).permute(0, 3, 1, 2)
    patch_x_tensor = TVT(patch_x_tensor)
    patch_Phi_tensor = TVT(patch_Phi_tensor)
    patch_x_tensor = patch_x_tensor.float()
    patch_Phi_tensor = patch_Phi_tensor.float()
    patch_g_tensor = A_torch(patch_x_tensor, patch_Phi_tensor)
    patch_f0_tensor = At_torch(patch_g_tensor, patch_Phi_tensor)
    patch_PhiPhiT_tensor = torch.sum(patch_Phi_tensor*patch_Phi_tensor, axis=1)
    patch_f0_tensor = patch_f0_tensor.float()
    patch_g_tensor = patch_g_tensor.float()
    patch_PhiPhiT_tensor = patch_PhiPhiT_tensor.float()
    return patch_x_tensor, patch_f0_tensor, patch_g_tensor, patch_Phi_tensor, patch_PhiPhiT_tensor
    

def train():
    batch_size = 64
    TOTAL_ITERS = 400000
    ITERS_PER_SAVE = 10 #Epochs per save
    #devices = [10,11,13,14]
    devices = [4,5]
    SAVE_PATH = './ckpts_ICVL'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    mse_loss = nn.MSELoss()
    model = DNU(31, K=21)
    TVT, TMO = set_devices(devices)
    model_w = DataParallel(model)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader, patch_num = get_dataloader(batch_size=batch_size)
    iter_per_ep = patch_num // batch_size
    total_ep = 160
    modules_optims = [model, optimizer]
    TMO(modules_optims)
    for ep in range(total_ep):
        if ep > 0 and  ep % 10 == 0:
            lr = lr * 0.9
            for g in optimizer.param_groups:
                g['lr'] = lr
        count = 0
        for patch, mask in dataloader:
            x, f0, g, Phi, PhiPhiT = convert_2_tensor(patch, mask, TVT)
            x = TVT(x)
            f0 = TVT(f0)
            g = TVT(g)
            Phi = TVT(Phi)
            PhiPhiT = TVT(PhiPhiT)
            dnu_out = model_w(f0, g, Phi, PhiPhiT)
            optimizer.zero_grad()
            loss = mse_loss(dnu_out, x)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 10 == 0:
                print('ep %03d iter %06d / %06d, lr=%.5f, loss=%.6f' % (ep+1, count, iter_per_ep, lr, loss))
        if (ep+1) % ITERS_PER_SAVE == 0:
            save_path = os.path.join(SAVE_PATH, 'ep_%03d.pth' % (ep+1) )
            torch.save(model.state_dict(), save_path)
         

if __name__ == '__main__':
    train()


















