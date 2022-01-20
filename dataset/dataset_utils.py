from fnmatch import translate
import torch
from PIL import Image
from torch.utils.data import DataLoader
import quaternion, os
import numpy as np
import cv2
import glob, open3d
from torchvision import transforms
import MinkowskiEngine as ME
from .utils import MinkLocParams
from .New7scenesDataloader import  New7Scene_dataset, TrainTransform, TrainSetTransform

import sys
sys.path.append("/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention")
from model.model_utils import make_model

def make_collate_fn(dataset: New7Scene_dataset, mink_quantization_size=0.01):

    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        img_depth = [e[0] for e in data_list]
        clouds    = [e[1] for e in data_list]
        translate_s  = [e[2] for e in data_list]
        quaternion_s = [e[3] for e in data_list]

        batch_imgdepth = torch.stack(img_depth, dim=0)
        batch_pcd = torch.stack(clouds, dim=0) 
        batch_T = torch.stack(translate_s, dim=0)
        batch_Q = torch.stack(quaternion_s, dim=0)

              # Produces (batch_size, n_points, 3) tensor
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            batch_pcd = dataset.set_transform(batch_pcd)

        if mink_quantization_size is None:
            # Not a MinkowskiEngine based model
            batch_pcd = {'cloud': batch_pcd}
        else:
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                      for e in batch_pcd]
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch_sparse = {'coords': coords, 'features': feats}

        return batch_imgdepth, batch_sparse, batch_T, batch_Q

    return collate_fn


def make_dataloaders(cfg):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(cfg)

    dataloders = {}

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  cfg.mink_quantization_size)
    dataloders['train'] = DataLoader(datasets['train'],  collate_fn=train_collate_fn,
                                     num_workers=cfg.num_workers, batch_size=cfg.batch_size,pin_memory=True)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    # Currently validation dataset has empty set_transform function, but it may change in the future
    val_collate_fn = make_collate_fn(datasets['val'], cfg.mink_quantization_size)
    dataloders['val'] = DataLoader(datasets['val'], collate_fn=val_collate_fn,
                                       num_workers=cfg.num_workers,batch_size=cfg.batch_size, pin_memory=True)

    return dataloders['train'],  dataloders['val']


def make_datasets(cfg):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(cfg.aug_mode)
    train_set_transform = TrainSetTransform(cfg.aug_mode)

    transform_depth  = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])

    datasets['train'] = New7Scene_dataset(data_dir = cfg.data_dir, scene = cfg.scene, seq_list= cfg.train_seq_list, transform_depth=transform_depth, transform_pcd=train_transform,  set_transform=train_set_transform)#

    datasets['val'] = New7Scene_dataset(data_dir = cfg.data_dir, scene = cfg.scene, seq_list= cfg.val_seq_list, transform_depth=transform_depth, )

    return datasets

