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
from mmdet3d.models import build_backbone
from torchvision.transforms.transforms import ToPILImage

import spconv

def model_parser(model='ResNet', fixed_weight=False, dropout_rate=0.0, bayesian=False):
    pass

    # if model == 'GoogleNet':
    #     base_model = models.inception_v3(pretrained=True)
    #     base_model.Conv2d_1a_3x3.conv = nn.Conv2d(1,32,kernel_size=3,stride=2,bias=False)
    #     # print(base_model)
    #     network = GoogleNet(base_model, fixed_weight, dropout_rate)

    # elif model == 'ResNet':
    #     base_model = models.resnet34(pretrained=True)
    #     base_model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    #     network = ResNet(base_model, fixed_weight, dropout_rate, bayesian)
    # elif model == 'ResnetSimple':
    #     base_model = models.resnet34(pretrained=True)
    #     network = ResNetSimple(base_model, fixed_weight)
    # else:
    #     assert 'Unvalid Model'

    #return network


def convert_pcd_to_spnet(input_pcd):
    #input_pcd : tensor type [batch_size, numpoint, feature]

    b_s , nun_p, vec_length = input_pcd.shape
    assert nun_p ==512 * 600
    pcd_x = input_pcd.reshape(b_s, 512, 600, 3 ) # batch_szie, num point, feature
    pcd_x_sp = spconv.SparseConvTensor.from_dense(pcd_x)

    return pcd_x_sp

class PoseLoss(nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -6.25

        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        # if learn_beta:
        #     self.sx.requires_grad = True
        #     self.sq.requires_grad = True
        #
        # self.sx = self.sx.to(device)
        # self.sq = self.sq.to(device)

        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        loss_x = F.l1_loss(pred_x, target_x)
        loss_q = F.l1_loss(pred_q, target_q)

        loss = torch.exp(-self.sx) * loss_x \
               + self.sx \
               + torch.exp(-self.sq) * loss_q \
               + self.sq

        #self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()

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
        x = torch.flatten(x,1)
        #x = x.view(x.size(0), -1)
        #x = self.fc_last(x)
        return x 

class PointNet2(nn.Module):
    def __init__(self, in_channels =3 ):
        super(PointNet2, self).__init__()

        self.cfg = dict(
            type='PointNet2SAMSG',
            in_channels=in_channels,  # [xyz, rgb]
            num_points=(1024, 256, 64, 16),
            radii=((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)),
            num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
            sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96,
                                                                        128)),
                         ((128, 196, 256), (128, 196, 256)), ((256, 256, 512),
                                                              (256, 384, 512))),
            aggregation_channels=(None, None, None, None),
            fps_mods=(('D-FPS'), ('D-FPS'), ('D-FPS'), ('D-FPS')),
            fps_sample_range_lists=((-1), (-1), (-1), (-1)),
            dilated_group=(False, False, False, False),
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type='BN2d'),
            sa_cfg=dict(
                type='PointSAModuleMSG',
                pool_mod='max',
                use_xyz=True,
                normalize_xyz=False))
        self.model = build_backbone(self.cfg)

    def forward(self, x):
        # print(x.shape)
        x = self.model(x)['sa_indices']
        
        sa_vector = torch.cat( x[1:], dim=1 ) # ( B, 1024+256+64+16 = 1360)

        return sa_vector


class SpConvNet(nn.Module):
    def __init__(self,in_channel=3 ):
        super().__init__()
        self.net = spconv.SparseSequential(

            spconv.SparseConv2d(in_channel, 64, 3,2), # just like nn.Conv3d but don't support group and all([d > 1, s > 1])
            nn.BatchNorm1d(64), # non-spatial layers can be used directly in SparseSequential.
            nn.ReLU(),

            spconv.SparseConv2d(64, 64, 5, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            spconv.SparseConv2d(64, 32, 5, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            spconv.SparseMaxPool2d(2, 2),
            spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.

            nn.Conv2d(32,16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x):
        #coors = coors.int() # unlike torch, this library only accept int coordinates.
        #x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        x = self.net(x)
        x =torch.flatten(x,1)
        return x # .dense()


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
    def __init__(self, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(Fuse_PNet, self).__init__()
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # 1. create pcd net
        self.Pcd_Net = PointNet2( in_channels =3 ).cuda() # out put (B , 1360)
        
        #     get fusion size
        pcd_out_channel = self.Pcd_Net( torch.randn(1,1,3).cuda() ).size(1)

        # 2. create depth net  
        self.Depth_Net = Depth_Net( pcd_out_channel )
        
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

        x_depth = self.Depth_Net(x_depth) # (B,1360)
        x_pcd = self.Pcd_Net(x_pcd) # (B,1360)

        fusion_vector = torch.cat( [x_depth, x_pcd], dim = 1 )
        att_out = self.attention( fusion_vector ) 

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            att_out = F.dropout(att_out, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(att_out)
        rotation = self.fc_rotation(att_out)

        return position, rotation

class Fuse_SPNet(nn.Module):
    def __init__(self, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(Fuse_SPNet, self).__init__()
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # 1. create pcd net
        # self.Pcd_Net = PointNet2( in_channel =3 ).cuda() # output: torch.Size([10, 15232])
        self.Pcd_Net  = SpConvNet( in_channel = 3).cuda()
        
        # get pcd size
        pcd_x = torch.randn(1, 512 * 600, 3).cuda()
        # 转为 NHWC 格式
        pcd_x = pcd_x.reshape(1, 512, 600, 3 ) # batch_szie, num point, feature
        pcd_x_sp = spconv.SparseConvTensor.from_dense(pcd_x)
        pcd_out_channel = self.Pcd_Net( pcd_x_sp ).size(1) 

        # 2. create depth net  
        self.Depth_Net = Depth_Net().cuda()
        # get depth size
        depth_channel = self.Depth_Net( torch.randn(1,1,224,224).cuda() ).size(1) 
        
        if fixed_weight:
            for param in self.Depth_Net.parameters():
                param.requires_grad = False

        # 3. create self attention
        fusion_vector_size = depth_channel + pcd_out_channel
        self.attention = AttentionBlock( fusion_vector_size )
            
        self.fc_position = nn.Linear( fusion_vector_size , 3, bias=True)
        self.fc_rotation = nn.Linear( fusion_vector_size , 4, bias=True)

        init_modules = [self.fc_position, self.fc_rotation]
        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, x_depth, x_pcd):

        x_depth = self.Depth_Net(x_depth) # (B,1360)
        x_pcd = self.Pcd_Net(x_pcd) # (B,1360)

        fusion_vector = torch.cat( [x_depth, x_pcd], dim = 1 )
        att_out = self.attention( fusion_vector ) 

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            att_out = F.dropout(att_out, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(att_out)
        rotation = self.fc_rotation(att_out)

        return position, rotation


####


def test_depth():
    print("x")
    x = torch.randn(10, 1, 224 , 224).cuda()
    D_Net = Depth_Net().cuda()
    x = D_Net(x)
    print(x.shape) #torch.Size([10, 512, 1, 1])

def test_pointnet():
    print("x")
    model = PointNet2().cuda()
    data = model( torch.randn(1,10,3).cuda() )#.size(1)
    print(data.shape) # torch.Size([1, 1360])

def test_spconv():
    print("x")
    #假设一次读取点云 # batch_zie ， num , feature 
    x = torch.randn(10, 256 * 256, 3).cuda()
    # 转为 NHWC 格式
    x = x.reshape(10, 256, 256, 3 ) # batch_szie, num point, feature
    x_sp = spconv.SparseConvTensor.from_dense(x)

    net = spconv.SparseSequential(
            nn.BatchNorm1d(3),
            spconv.SparseConv2d(3, 32, 3, 1),
            nn.ReLU(),
            spconv.SparseConv2d(32, 64, 3, 1),
            nn.ReLU(),
            spconv.SparseMaxPool2d(2, 2),
            spconv.ToDense(), 
        ).cuda()
    x = net(x_sp)
    print(x.shape)
    #data = model(  )#.size(1)
    #model = PointNet2().cuda()
    #print(data.shape)

def test_spconv2():
    #pass
    #假设一次读取点云 # batch_zie ， num , feature 
    x = torch.randn(10, 512 * 600, 3).cuda()
    # 转为 NHWC 格式
    x = x.reshape(10, 512, 600, 3 ) # batch_szie, num point, feature
    x_sp = spconv.SparseConvTensor.from_dense(x)
    
    model = SpConvNet(3).cuda()
    x = model(x_sp)
    print(x.shape) #torch.Size([10, 64, 123, 123])
    x = torch.flatten(x, 1)
    print(x.shape)

def test_spconv3():
    features =  torch.randn(256,6)# your features with shape [N, numPlanes]
    indices =2 # your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
    spatial_shape =16 # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
    batch_size =10 # batch size of your sparse tensor.
    x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
    x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
    print(x_dense_NCHW.sparity) # helper function to check sparity. 

def test_atten():
    x = torch.randn(10, 2360).cuda()
    attention = AttentionBlock( 2360 ).cuda()
    x =attention(x)
    print(x.shape)

def test_fusion():
    depth_x = torch.randn(10, 1, 224 , 224).cuda()
    pcd_x = torch.randn(10, 512 * 600, 3).cuda()
    # 转为 NHWC 格式
    pcd_x = pcd_x.reshape(10, 512, 600, 3 ) # batch_szie, num point, feature
    pcd_x_sp = spconv.SparseConvTensor.from_dense(pcd_x)

    fu_model = Fuse_SPNet().cuda()
    t1,q1 = fu_model(depth_x, pcd_x_sp )

    print(t1.shape, q1.shape)


if __name__=="__main__":
    #test_spconv2()
    #test_depth()
    #test_pointnet()
    #test_atten()
    test_fusion()
