

from PIL import Image
import torch


def norm_q(x_q_base):

    Norm = torch.norm(x_q_base, 2, 1)
    norm_q_base = torch.div(torch.t(x_q_base), Norm)
    
    return torch.t(norm_q_base)


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