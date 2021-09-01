# -*- coding: utf-8 -*-

from platform import machine
import torch
from torch.utils.data._utils import collate
from torchvision import datasets, transforms
import torch.utils.data as data
import os, sys
sys.path.append("./")
sys.path.append("../")
import math
import torch.nn as nn
from model import Fuse_SPNet, Fuse_PPNet, convert_pcd_to_spnet
# from model.ResNet50 import Res50PoseRess,Res50PoseRess_rgb
from dataset import data2d3d_loader
from mmcv import Config
from utils import median, norm_q
import logging, time


## step 1: config 
config_path = sys.argv[1]
assert os.path.exists(config_path)==True
config = Config.fromfile(config_path)
dtype = config.dtype # torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.cuda.manual_seed(1)

savedir = os.path.join(config.save_dir, config.save_prj, config.scene )  #'/media/fangxu/Disk4T/LQ/'+scene
if not os.path.exists(savedir):
    os.makedirs( savedir )

# step 2: logging setting, 输出到屏幕和日志
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
log_path = os.path.join( savedir, str(int( time.time() )) )
handler = logging.FileHandler(  )
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
 
logger.addHandler(handler)
logger.addHandler(console)




## step 2: data load
data_dir = os.path.join( config.data_dir , config.scene )
# label_dir_train = data_dir+'/singleImg/train.txt'
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
#1246
dataset_train = data2d3d_loader(data_dir,seq_list = [1,2,4,6], scene =config.scene , transform_depth =train_transform)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)

###

# label_dir_test = data_dir+'/singleImg/test.txt'
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset_test = data2d3d_loader(data_dir,scene =config.scene, seq_list = [3,5], transform_depth = test_transform)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_size,  shuffle=False)


## step 3: construct model 

if config.backbone =="ponint++":
    model = Fuse_PPNet()
elif config.backbone == "spconvnet":
    model = Fuse_SPNet()
else:
    logging.error("No model can be selected")
    assert False

criterion = nn.MSELoss()

criterion.cuda()
model.cuda()

adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)

loss_t_lst = []
loss_q_lst = []
loss_c_lst = []
loss_lst = []
median_lst = []

Best_Pos_error = 9999.0
Best_Ort_error = 9999.0

for e in range(config.epochs):
    logging.info('\n\nEpoch {} of {}'.format(e, config.epochs))

    model.train()

    loss_t_counter = 0.0
    loss_q_counter = 0.0
    loss_counter = 0.0
    t = 0
    for i, (img_base , downpcd_arr , base_t,base_q) in enumerate(train_loader):

        if i % config.print_every == 0:
            logging.info('Batch {} of {}'.format(i, len(train_loader)))
        # imgs_base = Variable(img_base.type(dtype))
        img_base = img_base.cuda()
        base_t  = base_t.cuda()
        base_q = base_q.cuda()
        # for i in downpcd_arr.size(0):
        downpcd_arr = downpcd_arr.cuda()
        downpcd_arr = convert_pcd_to_spnet(downpcd_arr)
        #downpcd_arr = torch.FloatTensor(downpcd_arr).cuda()

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
    logging.info('Average translation loss over epoch = {}'.format(loss_t_counter / (t + 1)))
    logging.info('Average orientation loss over epoch = {}'.format(loss_q_counter / (t + 1)))
    # print('Average content loss over epoch = {}'.format(loss_c_counter / (i + 1)))
    logging.info('Average loss over epoch = {}'.format(loss_counter / (t + 1)))

    pdist = nn.PairwiseDistance(2)
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
                #imgs_ba = Variable(img_base.type(dtype))

                imgs_ba = img_base.cuda()
                downpcd_arr = downpcd_arr.cuda()
                downpcd_arr = convert_pcd_to_spnet(downpcd_arr)

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
                logging.info(Best_Pos_error, Best_Ort_error)
                best_dict = os.path.join()
                save_best_path = os.path.exists(savedir, 'Best_params.pt')

                isExists = os.path.exists( save_best_path )
                if (isExists):
                    os.remove(save_best_path )
                torch.save(model.state_dict(), save_best_path )
            median_lst.append([dis_Err_i, ort2_Err_i])

            # print('average Distance err  = {} ,average orientation error = {} average Error = {}'.format(loss_counter / j,sum(dis_Err_Count)/j, sum(ort_Err_count)/j))
            logging.info('Media distance error  = {}, median orientation error2 = {}'.format(dis_Err_i, ort2_Err_i))
            logging.info(median_lst)

# if __name__=="__main__":

#     main()