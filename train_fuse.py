# -*- coding: utf-8 -*-

from platform import machine
import torch,math
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append("/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention")

from tqdm import tqdm
from dataset import make_dataloaders
from model import Fuse_PPNet, Pose_Depth_Net, standard_pose_loss

from mmcv import Config
from utils import median, quaternion_angular_error
import logging, time


## step 1: config
if len(sys.argv)==2:
    config_path = sys.argv[1]
else:
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

# if config.pretrain_weight is not None:
#     model.load_state_dict( torch.load(config.pretrain_weight) )

data_length = len(train_loader)
criterion = standard_pose_loss(config)
criterion.to(device)
model.to(device)

# error calculate
pdist = nn.PairwiseDistance(2)

adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)

Best_Pos_error = 9999.0
Best_Ort_error = 9999.0
t =  len( train_loader )
print(t)

for e in range(1,config.epochs+1):

    model.train()

    loss_t_acc = 0.0
    loss_q_acc = 0.0
    loss_acc = 0.0

    for i, (depth_base , pcd_base,  base_t,base_q) in enumerate(train_loader):

        depth_input = depth_base.to(device)
        pcd_input = {e: pcd_base[e].to(device) for e in pcd_base}
        t_target  = base_t.to(device)
        q_target = base_q.to(device)

        adam.zero_grad()

        t_pred, q_pred = model(depth_input, pcd_input)
        loss , loss_t_item, loss_q_item  = criterion(t_pred, q_pred, t_target, q_target)

        loss_t_acc = loss_t_acc + loss_t_item
        loss_q_acc = loss_q_acc + loss_q_item
        loss_acc += loss.item()

        loss.backward()
        adam.step()
 
        if i % config.print_every == 0:
            logger.info('epoch {}, batch:{}/{}, loss: {}, loss_T:{}, loss_Q:{} '.format(e, i , data_length ,
                                                             round(loss.item(), 5),
                                                             round(loss_t_item, 5),
                                                             round(loss_q_item, 5)   ))

    logger.info('Epoch:{}, Average translation loss over epoch = {}'.format(e, round( loss_t_acc/t , 5)  ))
    logger.info('Epoch:{}, Average orientation loss over epoch = {}'.format(e, round( loss_q_acc/t , 5)  ))
    logger.info('Epoch:{}, Average loss over epoch = {}'.format(e, round( loss_acc/t,5 )  ))

    if (e > -1 and e % config.interval == 0):

        model.eval()
        with torch.no_grad():
            dis_Err_Count = []
            ort_Err_count = []

            for i, (depth_base , pcd_base, base_t,base_q) in enumerate(tqdm(test_loader) ):

                depth_input = depth_base.to(device)
                pcd_input = {e: pcd_base[e].to(device) for e in pcd_base}
                t_gt = base_t.to(device)
                q_gt = base_q.to(device)

                t_infer, q_infer = model(depth_input, pcd_input)

                dis_Err = pdist(t_infer, t_gt).cpu().numpy()
                dis_Err_Count = dis_Err_Count + list(dis_Err)  # 合并为大 list , list + 号运算，非数值相加

                q_infer = F.normalize(q_infer, p=2, dim=1 )
                ort_Err = quaternion_angular_error( q_infer, q_gt).cpu().numpy()
                ort_Err_count = ort_Err_count + list(ort_Err) # 合并为大 list , list + 号运算，非数值相加
            
            pos_Err_e = median(dis_Err_Count)
            ort_Err_e = median(ort_Err_count)

            logger.info('Eval: Media distance error= {}, Median orientation error = {}'.format( round(pos_Err_e,5),
                                                                 round(ort_Err_e, 5) ))

            if pos_Err_e < Best_Pos_error:
                Best_Pos_error = pos_Err_e
                Best_Ort_error = ort_Err_e
                logger.info("{}, {}".format(Best_Pos_error, Best_Ort_error))
        
                save_best_path = os.path.join(savedir, 'Best_params_pcd_att_{}.pt'.format(e))
                logger.info('save the best params in epoch  {}'.format(e))
                #isExists = os.path.exists( save_best_path )
                #if (isExists):
                    #os.remove(save_best_path )
                torch.save(model.state_dict(), save_best_path )
# if __name__=="__main__":
#     print("x")

#     main()
