import os
import os.path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
import utils
import glob
import imageio
from PIL import Image
import h5py

DATASET_REGISTRY = {}


def build_dataset(name, *args, **kwargs): 
  return DATASET_REGISTRY[name](*args, **kwargs)


def register_dataset(name):
    def register_dataset_fn(fn):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = fn
        return fn

    return register_dataset_fn

@register_dataset("for_git")
def load_data(data_root, args, batch_size=32, num_workers=0, valid_size=0.05):

    data_path = os.path.join(data_root,args.h5_file)
    concat_mul = 10

    with h5py.File(data_path,'r') as f:
      # train_data = f['train']
      num_train = len(f['train'])#train_data.shape[0]
      num_test = len(f['test'])
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    
    pin_memory = False
    train_params = {'batch_size': args.batch_size,
          'num_workers': num_workers,
          'drop_last': False,
          'shuffle': True,
          'pin_memory':pin_memory
          }
    val_params = {'batch_size': args.batch_size,
          'num_workers': num_workers,
          'drop_last':False,
          'shuffle' : True,
          'pin_memory':pin_memory
          }
    test_params = {'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 1,
          'drop_last':False,
          'pin_memory':pin_memory
          }

    train_dataset = (data_path,'train')
    test_dataset = (data_path, 'test')

    train_set = utils.Dataset(train_dataset, args, idxs = train_idx)
    val_set = utils.Dataset(train_dataset, args, idxs = valid_idx)
    test_set = utils.Dataset(test_dataset, args, idxs = None, n_samples=num_test)

    train_set = data.ConcatDataset([train_set,]*concat_mul)
    
    train_loader = torch.utils.data.DataLoader(train_set,**train_params)
    valid_loader = torch.utils.data.DataLoader(val_set,**val_params)
    test_loader = torch.utils.data.DataLoader(test_set,**test_params)


    return train_loader, valid_loader, test_loader

@register_dataset("test_lf")
def load_data(data_root, args, batch_size=32, num_workers=0, valid_size=0.05):
    data_path = os.path.join(data_root,args.h5_file)

    with h5py.File(data_path,'r') as f:
      num_test = len(f['test'])
      print(f['test'].shape)
    
    test_params = {'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 1,
          'drop_last':False
          }

    test_dataset = (data_path, 'test')

    test_set = utils.TestLFDataset(test_dataset, args, idxs = None, n_samples=num_test)

    test_loader = torch.utils.data.DataLoader(test_set,**test_params)

    return test_loader


@register_dataset("test_st")
def load_data(data_root, args, batch_size=32, num_workers=0, valid_size=0.05):
    data_path = os.path.join(data_root,args.h5_file)

    with h5py.File(data_path,'r') as f:
      num_test = len(f['test'])
      print(f['test'].shape)
    
    test_params = {'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 1,
          'drop_last':False
          }

    test_dataset = (data_path, 'test')

    test_set = utils.TestStereoDataset(test_dataset, args, idxs = None, n_samples=num_test)

    test_loader = torch.utils.data.DataLoader(test_set,**test_params)

    return test_loader