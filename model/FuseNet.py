import os
import time
import copy
from typing import DefaultDict
import torch
# import torchvision
# import pandas as pd
# import numpy as np
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
#from mmdet3d.models import build_backbone
from torchvision.transforms.transforms import ToPILImage
from .model_utils import make_model 



# process RGB or depth I
class Depth_Net(nn.Module):
    def __init__(self, out_channels=10):
        super(Depth_Net, self).__init__()

        base_model = models.resnet34(pretrained=True)
        base_model.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=3,bias=False)

        feat_in = base_model.fc.in_features
        seq_net_list = list(base_model.children())[:-1]

        self.fc_last = nn.Linear(feat_in, out_channels , bias=True)
        # seq_net_list.append( self.fc_last )

        init_modules = [self.fc_last]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.resnet = nn.Sequential( *seq_net_list )
        # print(self.resnet)

    def forward(self, x):

        x = self.resnet(x)
        #x = torch.flatten(x,1)
        x = x.view(x.size(0), -1)
        x = self.fc_last(x)
        return x 


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        g_x = self.g(x).view(batch_size, out_channels // 8, 1)

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z

class Fuse_PPNet(nn.Module):
    def __init__(self,cfg=None, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(Fuse_PPNet, self).__init__()
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # 1. create pcd net
        self.Pcd_Net = make_model(cfg) # out put (B , 256)
        # model.load_state_dict( torch.load(config.pretrain_weight) )
        pcd_out_channel = cfg.output_dim 

        # 2. create depth net  
        self.Depth_Net = Depth_Net( out_channels = pcd_out_channel )
        
        if fixed_weight:
            for param in self.Depth_Net.parameters():
                param.requires_grad = False

        # 3. create self attention
        self.attention = AttentionBlock( pcd_out_channel*2 )
            
        self.fc_position = nn.Linear( pcd_out_channel*2 , 3, bias=True)
        self.fc_rotation = nn.Linear( pcd_out_channel*2 , 4, bias=True)
        
        init_modules = [self.fc_position, self.fc_rotation]
        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x_depth, x_pcd):

        x_depth = self.Depth_Net(x_depth)
        x_pcd = self.Pcd_Net(x_pcd)
        fusion_vector = torch.cat( [x_depth, x_pcd], dim = 1 )
        att_out = self.attention( fusion_vector )

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            att_out = F.dropout(att_out, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(att_out)
        rotation = self.fc_rotation(att_out)

        return position, rotation


class Pose_Depth_Net(nn.Module):
    def __init__(self, out_channels=10):
        super(Pose_Depth_Net, self).__init__()

        base_model = models.resnet34(pretrained=True)
        base_model.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=3,bias=False)

        feat_in = base_model.fc.in_features
        seq_net_list = list(base_model.children())[:-1]

        self.fc_last = nn.Linear(feat_in, out_channels , bias=True)
        # seq_net_list.append( self.fc_last )

        init_modules = [self.fc_last]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.resnet = nn.Sequential( *seq_net_list )
        # print(self.resnet)

    def forward(self, x):

        x = self.resnet(x)
        x = torch.flatten(x,1)
        #x = x.view(x.size(0), -1)
        x = self.fc_last(x)
        return x[:,:3],x[:,3:]
