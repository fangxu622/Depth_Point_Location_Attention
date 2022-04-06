

from PIL import Image
import torch
import numpy as np

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