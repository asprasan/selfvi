import torch
from torch.utils import data
import glob
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
import h5py
from train import get_args


class Dataset(data.Dataset):
    def __init__(self, imgs, args, idxs=None, ps=128, n_samples=None):
        self.frames = args.seq_len
        self.data = imgs[0]
        self.mode = imgs[1]
        # angular = args.angular
        # left_idx = int(angular * (angular // 2))
        # right_idx = int(left_idx + (angular - 1))
        # self.lf_view_idx = [left_idx, right_idx]
        self.idxs = range(n_samples) if idxs is None else idxs
        # self.list_IDs = imgs
        self.psh = args.inph
        self.psw = args.inpw

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.idxs)

    def __getitem__(self, index):
        'Generates one sample of data'
        max_shift = 10
        inputs = {}
        # pytorch gives error when using multiple workers with hdf5 loading
        # so, open the h5 file for each worker separately
        list_imgs = h5py.File(self.data, 'r')[self.mode]

        idx = self.idxs[index]
        I_np = list_imgs[idx]

        # take a random crop
        # but move the disparity plane around for the right image crop
        h, w = I_np.shape[3:]
        x = np.random.randint(0, h - self.psh - 2)
        y = np.random.randint(max_shift, w - self.psw - max_shift)
        # color = np.random.randint(3)

        # select a subset of frames for training
        k = np.random.randint(I_np.shape[0] - self.frames)

        shift = np.random.randint(-5, 2)
        color_scale = np.random.uniform(0.8, 1.0)

        left_img = I_np[k:k + self.frames, 0,
                        :, x:x + self.psh, y:y + self.psw]
        right_img = I_np[k:k + self.frames, 1, :, x:x +
                         self.psh, y - shift:y + self.psw - shift]
        cropped_stereo_pair = np.stack([left_img, right_img], 1)

        if np.random.randint(2):
            # do vertical flip
            cropped_stereo_pair = cropped_stereo_pair[..., ::-1, :].copy()
        inputs['video'] = torch.FloatTensor(
            color_scale * cropped_stereo_pair / 255.)
        return inputs


class TestLFDataset(data.Dataset):
    def __init__(self, imgs, args, idxs=None, ps=128, n_samples=None):
        self.frames = args.seq_len
        self.data = imgs[0]
        self.mode = imgs[1]
        self.u = self.v = args.angular
        # angular = args.angular
        # left_idx = int(angular * (angular // 2))
        # right_idx = int(left_idx + (angular - 1))
        # self.lf_view_idx = [left_idx, right_idx]
        self.idxs = range(n_samples) if idxs is None else idxs
        # self.list_IDs = imgs
        self.ps = ps
        self.psh = args.inph
        self.psw = args.inpw

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.idxs)

    def __getitem__(self, index):
        'Generates one sample of data'
        inputs = {}

        list_imgs = h5py.File(self.data, 'r')[self.mode]

        idx = self.idxs[index]
        I_np = list_imgs[idx]
        
        h,w = I_np.shape[-2:]
        k = np.random.randint(I_np.shape[0] - self.frames)

        I_np = I_np[:self.frames]
        gt_angular = int(np.sqrt(list_imgs.shape[2]))
        # angular resolutioon of the gt video can be higher than that of the required video
        # e.g. Lytro gives 8x8 LF video, but we are generating only 5x5 videos
        # so, we crop the required number of views from the ground truth video
        I_np = I_np.reshape([self.frames,gt_angular,gt_angular,3,h,w])[:,:self.u,:self.v,...]
        I_np = I_np.reshape([self.frames,self.u*self.v,3,h,w])
        
        inputs['video'] = torch.FloatTensor(I_np/255.)
        return inputs