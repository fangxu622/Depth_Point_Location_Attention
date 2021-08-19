# -*- coding: utf-8 -*-

import torch
from torch.utils.data._utils import collate
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torch.utils.data as data
import os
import sys
sys.path.append("./")
sys.path.append("../")
import math
import torch.nn as nn
from model import fusion_model
# from model.ResNet50 import Res50PoseRess,Res50PoseRess_rgb
from dataset import data2d3d_loader

def norm_q(x_q_base):

    Norm = torch.norm(x_q_base, 2, 1)
    norm_q_base = torch.div(torch.t(x_q_base), Norm)

    return torch.t(norm_q_base)


def default_loader(path):
    return Image.open(path).convert('I')

# def default_loader(path):
#     return Image.open(path).convert('RGB')

def median(lst):
    lst.sort()
    if len(lst) % 2 == 1:
        return lst[len(lst) // 2]
    else:
        return (lst[len(lst) // 2 - 1]+lst[len(lst) // 2]) / 2.0

def my_collte_fn(batch):
    #for _,pcd,
    depth_data     = torch.stack([item[0] for item in batch])
    pcd_data       = [item[1] for item in batch]  # each element is of size (1, h*, w*). where (h*, w*) changes from mask to another.
    translate_data = torch.stack([item[2] for item in batch])
    qq_data        = torch.stack([item[3] for item in batch])
    
    return depth_data, pcd_data,translate_data,qq_data


learning_rate = 1e-4
batch_size =1
epochs = 500
cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print_every = 32

torch.cuda.manual_seed(1)

scene = 'chess'
data_dir = '/media/fangxu/Disk4T/LQ/data/'+scene
# label_dir_train = data_dir+'/singleImg/train.txt'
prep_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
#1246
dataset_train = data2d3d_loader(data_dir,seq_list = [1,2,4,6], scene ="chess",transform_depth =prep_train_transform)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

###

# label_dir_test = data_dir+'/singleImg/test.txt'
prep_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset_test = data2d3d_loader(data_dir,scene ="chess",seq_list = [3,5], transform_depth=prep_test_transform)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1,  shuffle=False)

model = fusion_model()

criterion = nn.MSELoss()

if cuda:
    criterion.cuda()
    model.cuda()

adam = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)


loss_t_lst = []
loss_q_lst = []
loss_c_lst = []
loss_lst = []
median_lst = []

Best_Pos_error = 9999.0
Best_Ort_error = 9999.0

for e in range(epochs):
    print('\n\nEpoch {} of {}'.format(e, epochs))

    model.train()

    loss_t_counter = 0.0
    loss_q_counter = 0.0
    loss_counter = 0.0
    t = 0
    for i, (img_base , downpcd_arr , base_t,base_q) in enumerate(train_loader):

        if i % print_every == 0:
            print('Batch {} of {}'.format(i, len(train_loader)))
        # imgs_base = Variable(img_base.type(dtype))
        img_base = img_base.cuda()
        base_t  = base_t.cuda()
        base_q = base_q.cuda()
        # for i in downpcd_arr.size(0):


        downpcd_arr = torch.FloatTensor(downpcd_arr).cuda()

        adam.zero_grad()
        x_t_base, x_q_base = model(img_base,downpcd_arr)

        norm_q_base = norm_q(x_q_base)

        loss_t = criterion(x_t_base, base_t)
        loss_q = criterion(x_q_base, base_q)

        loss_t_counter = loss_t_counter+loss_t.data
        loss_q_counter = loss_q_counter+loss_q.data
        loss = loss_t + loss_q


        loss_counter += loss.data

        loss.backward()
        adam.step()
        t = t+1
    print('Average translation loss over epoch = {}'.format(loss_t_counter / (t + 1)))
    print('Average orientation loss over epoch = {}'.format(loss_q_counter / (t + 1)))
    # print('Average content loss over epoch = {}'.format(loss_c_counter / (i + 1)))
    print('Average loss over epoch = {}'.format(loss_counter / (t + 1)))

    pdist = nn.PairwiseDistance(2)
    savedir = '/media/fangxu/Disk4T/LQ/'+scene
    isExists = os.path.exists(savedir)
    # if (e % 10 == 0):
    #     if(not isExists):
    #         os.mkdir(savedir)
    #
    #     torch.save(model.state_dict(), ('/media/fangxu/Disk4T/LQ/depth/'+scene+'/fuse2d3d_params_epoch{}.pt').format(e))

    if (e > -1 and e % 10 == 0):

        model.eval()
        with torch.no_grad():

            dis_Err_Count = []

            ort2_Err_count = []

            loss_counter = 0.

            for i, (img_base , downpcd_arr , base_t,base_q) in enumerate(train_loader):
                imgs_ba = Variable(img_base.type(dtype))

                downpcd_arr = torch.FloatTensor(downpcd_arr).cuda()

                x_t_base, x_q_base = model(imgs_ba,downpcd_arr)

                base_t = base_t.cuda()
                base_q = base_q.cuda()

                dis_Err = pdist(x_t_base, base_t)
                dis_Err_Count.append(float(dis_Err))

                x_q_base = norm_q(x_q_base)

                Ort_Err2 = float(2 * torch.acos(torch.abs(torch.sum(base_q * x_q_base, 1))) * 180.0 / math.pi)

                ort2_Err_count.append(Ort_Err2)
                # result.append([dis_Err,Ort_Err2])

            dis_Err_i = median(dis_Err_Count)
            ort2_Err_i = median(ort2_Err_count)

            if dis_Err_i < Best_Pos_error:
                Best_Pos_error = dis_Err_i
                Best_Ort_error = ort2_Err_i
                print(Best_Pos_error, Best_Ort_error)
                isExists = os.path.exists('/media/fangxu/Disk4T/LQ/depth/'+scene + '_Best_poseNetLSTM_params.pt')
                if (isExists):
                    os.remove('/media/fangxu/Disk4T/LQ/depth/'+scene + '_Best_poseNetLSTM_params.pt')
                torch.save(model.state_dict(),
                           '/media/fangxu/Disk4T/LQ/depth/'+scene + '_Best_poseNetLSTM_params.pt')
            median_lst.append([dis_Err_i, ort2_Err_i])

            # print('average Distance err  = {} ,average orientation error = {} average Error = {}'.format(loss_counter / j,sum(dis_Err_Count)/j, sum(ort_Err_count)/j))
            print('Media distance error  = {}, median orientation error2 = {}'.format(dis_Err_i, ort2_Err_i))
            print(median_lst)
