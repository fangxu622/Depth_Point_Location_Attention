
#from platform import machine
import torch,math
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append("/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention")

from tqdm import tqdm
from dataset import make_dataloaders
from model import Fuse_PPNet, Pose_Depth_Net, Pose_Pcd_Net , standard_pose_loss
from mmcv import Config

import logging, time
import numpy as np
torch.cuda.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_path = '/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/config/conf_pcd.py'
assert os.path.exists(config_path)==True
cfg = Config.fromfile(config_path)

# Pcd_Net = make_model(cfg).to(device)
# Pcd_Net.train()
model =  Pose_Pcd_Net(cfg)
model.to(device)

#model.train()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5)

train_loader , test_loader = make_dataloaders(cfg)
#i, (depth_base , pcd_base,  base_t,base_q) = enumerate(train_loader).__next__()
for i, ( _ , pcd_base,  base_t,base_q) in enumerate(train_loader):
    pcd_input = {e: pcd_base[e].to(device) for e in pcd_base}
    t_target, q_target  = base_t.to(device), base_q.to(device)

    print(t_target.shape)

    t_pred, q_pred = model(pcd_input)

    print(q_pred.shape)
