

from PIL import Image
import torch
import numpy as np
from collections import OrderedDict

# def quaternion_angular_error(q1, q2):
#     d = abs(np.dot(q1, q2))
#     d = min(1.0, max(-1.0, d))
#     theta = 2 * np.arccos(d) * 180 / np.pi
#     return theta

def quaternion_angular_error(q1,q2):
    d = torch.abs( torch.mul(q1,q2).sum(1)  )
    d[d>1.0] = 1.0
    theta = 2 * torch.acos( d ) * 180.0 /np.pi
    return theta


def median(lst):
    lst.sort()
    if len(lst) % 2 == 1:
        return lst[len(lst) // 2]
    else:
        return (lst[len(lst) // 2 - 1]+lst[len(lst) // 2]) / 2.0

def load_state_dict(model, state_dict):
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

  # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

def default_loader(path):
    #return Image.open(path).convert('RGB')
    return Image.open(path).convert('I')


def my_collte_fn(batch):
    #for _,pcd,
    depth_data     = torch.stack([item[0] for item in batch])
    pcd_data       = [item[1] for item in batch]  # each element is of size (1, h*, w*). where (h*, w*) changes from mask to another.
    translate_data = torch.stack([item[2] for item in batch])
    qq_data        = torch.stack([item[3] for item in batch])

    return depth_data, pcd_data,translate_data,qq_data