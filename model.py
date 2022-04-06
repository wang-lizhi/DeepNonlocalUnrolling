import math
import numpy as np
import torch
import scipy.io as sio
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torchvision
from utils import *
import os


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight.data, gain=1.0)
        #init.normal_(m.bias.data, mean=0.0, std=0.05)
        #init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)



# the spectral prior network S in Eq.16
# n_channels: input spectral channels 
# L: channel number of middle features
class MultiScale(nn.Module):
	def __init__(self,in_ch,out_ch1,mid_ch13,out_ch13,mid_ch15,out_ch15,out_ch_pool_conv):
		super(MultiScale,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_ch,out_ch1,kernel_size=1,stride=1).apply(weights_init_kaiming),
			nn.ReLU())
		self.conv13 = nn.Sequential(
			nn.Conv2d(in_ch,mid_ch13,kernel_size=1,stride=1).apply(weights_init_kaiming),
			nn.ReLU(),
			nn.Conv2d(mid_ch13,out_ch13,kernel_size=3,stride=1,padding=1).apply(weights_init_kaiming),
			nn.ReLU())
		
		self.conv15 = nn.Sequential(
			nn.Conv2d(in_ch,mid_ch15,kernel_size=1,stride=1).apply(weights_init_kaiming),
			nn.ReLU(),
			nn.Conv2d(mid_ch15,out_ch15,kernel_size=5,stride=1,padding=2).apply(weights_init_kaiming),
			nn.ReLU())

		self.pool_conv1 = nn.Sequential(
			nn.MaxPool2d(3,stride=1,padding=1),
			nn.Conv2d(in_ch,out_ch_pool_conv,kernel_size=1,stride=1).apply(weights_init_kaiming),
			nn.ReLU())

	def forward(self,inputs,train=False):
		conv1_out = self.conv1(inputs)
		conv13_out = self.conv13(inputs)
		conv15_out = self.conv15(inputs)
		pool_conv_out = self.pool_conv1(inputs)
		outputs = torch.cat([conv1_out,conv13_out,conv15_out,pool_conv_out],1) # depth-wise concat
		return outputs

class SpectralPriorNet(nn.Module):
    def __init__(self, n_channels, L=64):
        super(SpectralPriorNet, self).__init__()
        # local branch layers
        self.stem_layer = nn.Sequential(
            nn.Conv2d(31,64,7,1,3).apply(weights_init_kaiming),
            nn.ReLU(),
            nn.Conv2d(64,64,1,1,0).apply(weights_init_kaiming),
            nn.ReLU(),
            nn.Conv2d(64,192,3,1,1).apply(weights_init_kaiming),
            nn.ReLU()
            )
#in_ch,out_ch_1,mid_ch_13,out_ch_13,mid_ch_15,out_ch_15,out_ch_pool_conv
        self.MultiScale_layer1 =MultiScale(192,64,96,128,16,32,32)
        self.l_conv = nn.Conv2d(in_channels=256,out_channels=n_channels,kernel_size=1,padding=0,bias=False).apply(weights_init_kaiming)
        self.local_prior_branch = nn.Sequential(
        nn.Conv2d(n_channels, L, kernel_size=3, padding=1, bias=False).apply(weights_init_kaiming),
        nn.ReLU(inplace=True),
        nn.Conv2d(L, n_channels, kernel_size=3, padding=1, bias=False).apply(weights_init_kaiming)
        )
        # non local branch layers
        self.nl_conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=1, padding=0, bias=False).apply(weights_init_kaiming)
        self.nl_relu = nn.ReLU(inplace=True)
        # adaptive parameter omega
        self.omega = torch.nn.Parameter(torch.FloatTensor([0.95])) #0.999
      


    def forward(self, x):
        [bs, nC, row, col] = x.shape
        # local branch feat
        local_branch = self.stem_layer(x)
        local_branch = self.MultiScale_layer1(local_branch)
        local_branch = self.l_conv(local_branch)
        local_branch_feat = self.local_prior_branch(local_branch)
        local_branch_feat = local_branch + local_branch_feat
        # calculate feat dist in non-local branch
        x_flat = x.view(bs, nC, -1) # [bs, nC, row*col]
        nl_feat = self.nl_conv1(x) # psi(f) in Eq.18
        # [bs, row*col, nC]
        nl_feat_flat = nl_feat.view(bs, nC, -1).permute(0,2,1).contiguous() 
        # get f^hat in Eq.18
        # f_hat = x_flat.T x f_flat x nl_feat_flat
        # instead of calculating  f_hat = (x_flat.T x f_flat) x nl_feat_flat, i.e.
        # [bs, row*col, nC] x [bs, nC, row*col] = [bs, row*col, row*col]
        # and [bs, row*col, row*col] x [bs, row*col, nC] = [bs, row*col, nC]
        # we note that this is an A x B x C matrix multiplication, so we can use A x (B x C)
        # to save memory
        t_var_1 = torch.matmul(x_flat, nl_feat_flat)
        f_hat = torch.matmul(x_flat.permute(0, 2, 1).contiguous(), t_var_1)
        # we will use the same matrix multiplication trick to normalize the weighted avg result
        #ones = torch.ones((bs, row * col, 1))
        #ones = ones.to(nl_feat_flat.device)
        #t_var_2 = torch.matmul(torch.transpose(nl_feat_flat, 1, 2), ones)
        #norm_term = torch.matmul(nl_feat_flat, t_var_2)
        #f_hat = f_hat / norm_term
        f_hat = f_hat * ( 1.0 / ((48 + 30) * 48 ))
        f_hat = f_hat.permute(0,2,1).contiguous().view(bs, nC, row, col)
        f_hat = self.nl_relu(f_hat)
        # now merge two branches and original x
        return self.omega * local_branch_feat + ( (1.0 - self.omega) * f_hat )
      

# one stage of DNU
# include matrix operation and prior network S to solve Eq.16
class DNU_stage(nn.Module):
    def __init__(self, n_channels, L=64, k=1):
        super(DNU_stage, self).__init__()
        self.S = SpectralPriorNet(n_channels, L)
        # adaptive parameter eta
        self.deta = torch.nn.Parameter(torch.FloatTensor([0.04]))
        self.eta = torch.nn.Parameter(torch.FloatTensor([0.8]))
    # f output from previous stage, for stage 0 it is inited as PhiT*g, shape=[bs, nC, row, col]
    # g the measurement matrix [bs, row, col]
    # Phi the system forward matrix, which is shifted and has the same shape as f
    # PhiPhiT is diag{phi_1,...phi_MN} as in paper
    # here we first take the diagonal as a vector, so its shape is [bs, row * col]
    # then we reshape it back to [bs, row, col]
    def forward(self, f, f0, g, Phi, PhiPhiT):
        [bs, nC, row, col] = f.shape
        # Eq.10
        h = self.S(f)
        # t3 is the term in the '[ ]' of Eq.16
        t1 = A_torch(f, Phi) # [bs, row, col]
        t2 = At_torch(t1, Phi) # [bs, nC, row, col]
        output = (1 - self.deta * self.eta) * f - self.deta * t2 + self.deta * f0 + self.deta * self.eta * h
        #output = self.sigmoid(output)
        return output


# the complete DNU, contains K stages
class DNU(nn.Module):
    def __init__(self, n_channels, L=64, K=11):
        super(DNU, self).__init__()
        self.L = L
        self.K = K
        stages = []
        for i in range(K):
            stages.append(DNU_stage(n_channels, L))
        self.stages = nn.ModuleList(stages)


    def forward(self, f, g, Phi, PhiPhiT):
        f0 = f.detach().clone()
        for i in range(self.K):
            f = self.stages[i](f, f0, g, Phi, PhiPhiT)
        return f


def test():
    import pdb
    pdb.set_trace()
    dnu = DNU(28)
    print(dnu)
    data_path = '/home/gmy/system/multispectral/CASSI-Self-Supervised-main'
    sample = '01'
    maskfile = data_path + '/Data/mask/mask_3d_shift.mat'
    r, c, nC = 256, 256, 28
    Phi = sio.loadmat(maskfile)['mask_3d_shift']
    index = int(sample)
    datapath = data_path + '/Data/kaist_data/scene'+ sample + '.mat'
    X_ori = sio.loadmat(datapath)['img']
    X_ori = X_ori/X_ori.max()
    X_ori_shift = shift(X_ori, step=2)
    
    patch_x = X_ori[54:54+48, 0:48,:]
    patch_Phi = Phi[54:54+48, 0:48,:]
    patch_g = A(patch_x, patch_Phi)
    patch_f0 = At(patch_g, patch_Phi)
    patch_PhiPhiT = np.sum(patch_Phi**2,2)

    patch_x_tensor = torch.zeros((1, nC, r, c))
    patch_Phi_tensor = torch.zeros((1, nC, r, c))
    patch_g_tensor = torch.zeros((1, r, c))
    patch_f0_tensor = torch.zeros((1, nC, r, c))
    patch_PhiPhiT_tensor = torch.zeros((1, nC, r, c))

    r, c, nC = 48, 48, 28
    patch_x_tensor = torch.from_numpy(patch_x).view(1, r, c, nC).permute(0, 3, 1, 2)
    patch_Phi_tensor = torch.from_numpy(patch_Phi).view(1, r, c, nC).permute(0, 3, 1, 2)
    patch_g_tensor = torch.from_numpy(patch_g).view(1, r, c)
    patch_f0_tensor = torch.from_numpy(patch_f0).view(1, r, c, nC).permute(0, 3, 1, 2)
    patch_PhiPhiT_tensor = torch.from_numpy(patch_PhiPhiT).view(1, r, c)
   
    dnu_out = dnu(patch_f0_tensor, patch_g_tensor, patch_Phi_tensor, patch_PhiPhiT_tensor)
    pass



if __name__ == '__main__':
    test()



