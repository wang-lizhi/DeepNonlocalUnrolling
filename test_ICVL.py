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
from datasets import HavardDataset_Test as HavardDataset
from model import DNU


def get_dataloader(batch_size=64):
    base_path = '/ICVL/test'
    data_names = os.listdir(base_path)
    data_paths = [ os.path.join(base_path, i) for i in mat_names if '.mat' in i]
    mask_path = '/ICVL/Cu_48.mat'
    datasets = [HavardDataset(mat_paths[i], mask_path) for i in range(len(mat_paths))]
    #print(len(hd))
    train_dataloaders = [DataLoader(datasets[i], batch_size=batch_size, shuffle=False, drop_last=False) for i in range(len(data_paths))]
    return train_dataloaders, datasets

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
    

def test():
    batch_size = 11
    TOTAL_ITERS = 400000
    ITERS_PER_SAVE = 1
    CRITIC_ITERS = [150000, 300000]
    #devices = [1
    devices = [3,]
    SAVE_PATH = './results_adam_tf_ICVL'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    mse_loss = nn.MSELoss()
    model = DNU(31, K=11)
    #model.load_state_dict(torch.load('./ckpts_omega0.95/ep_150.pth'))
    #model.load_state_dict(torch.load('./ckpts_tf_init_adam_tf/ep_160.pth'))
    #model.load_state_dict(torch.load('./ckpts_ICVL/ep_160.pth'))
    #model.load_state_dict(torch.load('./ckpts_adam_tf_ICVL/ep_160.pth'))
    model.load_state_dict(torch.load('./ckpts_ICVL/ep_160.pth'))
    model.eval()
    TVT, TMO = set_devices(devices)
    model_w = DataParallel(model)
    #dnu = dnu.cuda()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloaders, datasets = get_dataloader(batch_size=batch_size)
    modules_optims = [model, optimizer]
    TMO(modules_optims)
    
    base_path = '/ICVL/test'
    file_names = os.listdir(base_path)
    data_names = [ i for i in file_names if '.mat' in i]
    i = 0
    with torch.no_grad():
        for dataloader in dataloaders:
            count = 0
            #print('-------------------', mat_id[i], '-------------------')
            print('--------------------------------------')
            rec_patch = None
            for patch, mask in dataloader:
                x, f0, g, Phi, PhiPhiT = convert_2_tensor(patch, mask, TVT)
                x = TVT(x)
                f0 = TVT(f0)
                g = TVT(g)
                Phi = TVT(Phi)
                PhiPhiT = TVT(PhiPhiT)
                dnu_out = model_w(f0, g, Phi, PhiPhiT)
                if rec_patch is None:
                    rec_patch = dnu_out.cpu()
                else:
                    rec_patch = torch.cat((rec_patch, dnu_out.cpu()), 0)
                loss = mse_loss(dnu_out, x)
                count += 1
                if count % 1 == 0:
                    print('loss=%.6f' % (loss))
            rec_patch = rec_patch.permute(2, 3, 1, 0).cpu().numpy() # N, h, w, c
            sio.savemat(os.path.join(SAVE_PATH, data_names[i]), {'output':rec_patch, 'label': datasets[i].hyper})
            i += 1
         

if __name__ == '__main__':
    test()


















