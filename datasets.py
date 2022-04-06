import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
from utils import shift
import hdf5storage

class HavardDataset(Dataset):
    def __init__(self, mat_path=None, mask_path=None, patch_size=48):
        # load mask
        self.nC = 31
        self.patch_size = patch_size
        matfile = hdf5storage.loadmat(mat_path)
        self.patch = matfile['label'] # 48,48,nC,N

        #matfile = sio.loadmat(mask_path)
        matfile = hdf5storage.loadmat(mask_path)
        #matfile = h5py.File(mask_path, 'r')
        self.mask = matfile['Cu'] # 48,48,nC

        
    
    def __len__(self):
        return self.patch.shape[-1]

    # get a patch
    def __getitem__(self, idx):
        patch = self.patch[:,:,:,idx]
        mask = self.mask
        return patch, mask



class HavardDataset_Test(Dataset):
    def __init__(self, mat_path=None, mask_path=None, patch_size=48):
        # load mask
        self.nC = 31
        self.patch_size = patch_size
        matfile = hdf5storage.loadmat(mat_path)
        self.patch = matfile['patch_image'] # 48,48,nC,N
        self.hyper = matfile['hyper_image'] # 582x512x31

        #matfile = sio.loadmat(mask_path)
        matfile = hdf5storage.loadmat(mask_path)
        #matfile = h5py.File(mask_path, 'r')
        self.mask = matfile['Cu'] # 48,48,nC

        
    
    def __len__(self):
        return self.patch.shape[-1]

    # get a patch
    def __getitem__(self, idx):
        patch = self.patch[:,:,:,idx]
        mask = self.mask
        return patch, mask

class KAIST_Dataset(Dataset):
	def __init__(self,mask_path=None,HSI=None,num=40000):
		self.size = 96
		self.num = num
		self.HSI = HSI

        ## load mask
		data = hdf5storage.loadmat(mask_path)
		self.mask = data['mask']
		#self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 38))

	def __getitem__(self, index):
		index1   = random.randint(0, 30)
		hsi  =  self.HSI[:,:,:,index1]
        ## image patch
		
		shape = np.shape(hsi)
		px = random.randint(0, shape[0] - self.size)
		py = random.randint(0, shape[1] - self.size)
		label = hsi[px:px + self.size:1, py:py + self.size:1, :]

        ## mask patch
		pxm = random.randint(0, 512 - self.size)
		pym = random.randint(0, 512 - self.size)
		mask_3d = self.mask[px:px + self.size:1, py:py + self.size+37:1, :]


		rotTimes = random.randint(0, 3)
		vFlip    = random.randint(0, 1)
		hFlip    = random.randint(0, 1)
            # Random rotation
		for j in range(rotTimes):
			label  =  np.rot90(label)

            # Random vertical Flip
		for j in range(vFlip):
			label = label[:, ::-1, :].copy()

            # Random horizontal Flip
		for j in range(hFlip):
			label = label[::-1, :, :].copy()

		label = torch.FloatTensor(label.copy()).permute(2,0,1)
		return label, mask_3d

	def __len__(self):
		return self.num

class KAISTDataset_Test(Dataset):
    def __init__(self, mat_path=None, mask_path=None, patch_size=128):
        # load mask
        self.patch_size = patch_size
        matfile = hdf5storage.loadmat(mat_path)
        self.patch = matfile['image'] # 1*256*256*28
        #matfile = sio.loadmat(mask_path)
        matfile = hdf5storage.loadmat(mask_path)
        #matfile = h5py.File(mask_path, 'r')
        self.mask = matfile['mask'] # 1*256*256*28

        
    
    def __len__(self):
        return self.patch.shape[0]

    # get a patch
    def __getitem__(self, idx):
        patch = self.patch[:,:,:,idx]
        mask = self.mask
        return patch, mask


