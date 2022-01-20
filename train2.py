# -*- coding: utf-8 -*-

from platform import machine
import torch,math
import torch.nn as nn
import os, sys
sys.path.append("/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention")

from dataset import make_dataloaders
from model import Fuse_PPNet

from mmcv import Config
from utils import median, norm_q
import logging, time


## step 1: config 
config_path = '/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/config/conf_1.py'
assert os.path.exists(config_path)==True
config = Config.fromfile(config_path)
dtype = config.dtype # torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.cuda.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
savedir = os.path.join(config.save_dir, config.save_prj, config.scene )  #'/media/fangxu/Disk4T/LQ/'+scene
if not os.path.exists(savedir):
    os.makedirs( savedir )

# step 2: logging setting, 输出到屏幕和日志
logger = logging.getLogger()
logger.setLevel(level = logging.INFO)
log_path = os.path.join( savedir, str(int( time.time() ))+".log" )
handler = logging.FileHandler( log_path )
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)  
#logger.addHandler(console)

## step 2: data load and model construct

train_loader , test_loader = make_dataloaders(config)
model = Fuse_PPNet(config)

criterion = nn.MSELoss()
criterion.to(device)
model.to(device)

adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)

loss_t_lst = []
loss_q_lst = []
loss_c_lst = []
loss_lst = []
median_lst = []

Best_Pos_error = 9999.0
Best_Ort_error = 9999.0

for e in range(config.epochs):
    #logging.info('\n\nEpoch {} of {}'.format(e, config.epochs))

    model.train()

    loss_t_counter = 0.0
    loss_q_counter = 0.0
    loss_counter = 0.0
    t = 0
    for i, (img_base , pcd_base , base_t,base_q) in enumerate(train_loader):

        img_base = img_base.to(device)
        pcd_base = {e: pcd_base[e].to(device) for e in pcd_base}
        base_t  = base_t.to(device)
        base_q = base_q.to(device)

        adam.zero_grad()
        x_t_base, x_q_base = model(img_base,pcd_base)

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

        if i % config.print_every == 0:
            logger.info('epoch {}, batch:{}/{}, loss: {}'.format(e, i ,len(train_loader), loss.data ) )

    logger.info('Epoch:{}, Average translation loss over epoch = {}'.format(e, loss_t_counter / (t + 1)))
    logger.info('Epoch:{}, Average orientation loss over epoch = {}'.format(e, loss_q_counter / (t + 1)))
    # print('Average content loss over epoch = {}'.format(loss_c_counter / (i + 1)))
    logger.info('Epoch:{}, Average loss over epoch = {}'.format(e, loss_counter / (t + 1)))

    pdist = nn.PairwiseDistance(2)

    if (e > -1 and e % 10 == 0):

        model.eval()
        with torch.no_grad():
            dis_Err_Count = []
            ort2_Err_count = []
            loss_counter = 0

            for i, (img_base , pcd_base , base_t,base_q) in enumerate(test_loader):

                img_base = img_base.to(device)
                pcd_base = {e: pcd_base[e].to(device) for e in pcd_base}
                base_t = base_t.to(device)
                base_q = base_q.to(device)

                x_t_infer, x_q_infer = model(img_base,pcd_base)

                dis_Err = pdist(x_t_infer, base_t)
                dis_Err_Count.append(float(dis_Err))

                x_q_base = norm_q(x_q_base)

                Ort_Err2 = float(2 * torch.acos(torch.abs(torch.sum(base_q * x_q_infer, 1))) * 180.0 / math.pi)
                ort2_Err_count.append(Ort_Err2)
                # result.append([dis_Err,Ort_Err2])

            dis_Err_i = median(dis_Err_Count)
            ort2_Err_i = median(ort2_Err_count)

            if dis_Err_i < Best_Pos_error:
                Best_Pos_error = dis_Err_i
                Best_Ort_error = ort2_Err_i
                logger.info("{}, {}".format(Best_Pos_error, Best_Ort_error))

                save_best_path = os.path.join(savedir, 'Best_params_pcd_att.pt')
                isExists = os.path.exists( save_best_path )
                if (isExists):
                    os.remove(save_best_path )
                torch.save(model.state_dict(), save_best_path )
            median_lst.append([dis_Err_i, ort2_Err_i])

            # print('average Distance err  = {} ,average orientation error = {} average Error = {}'.format(loss_counter / j,sum(dis_Err_Count)/j, sum(ort_Err_count)/j))
            logger.info('Media distance error  = {}, median orientation error2 = {}'.format(dis_Err_i, ort2_Err_i))
            logger.info( str(median_lst) )

# if __name__=="__main__":
#     print("x")

#     main()