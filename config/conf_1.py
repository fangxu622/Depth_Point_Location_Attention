
from typing import Optional
import torch

#cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# datasetting
data_dir = "/media/fangxu/Disk4T/LQ/data/"
scene = "chess" #optional "chess", 
train_seq_list = [1,2,4,6]#
val_seq_list = [3,5]
aug_mode = 1 
mink_quantization_size = 0.01
num_workers = 8
input_type = [ "depth","pcd" ]


# Model setting
feature_size = 256
output_dim =  256      # Size of the final descriptor

#DepthNet_out_dim = 256
# Size of the local features from backbone network (only for MinkNet based models)
# For PointNet-based models we always use 1024 intermediary features
planes = [32,64,64]
layers = 1,1,1
num_top_down = 1
conv0_kernel_size = 5

pretrain_weight =None #"/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/Res/experiment_1/chess/Best_params_pcd_att_90.pt"

model_name="MinkFPN_GeM"

## optimazation setting
learning_rate = 5e-5
train_batch_size =25
val_batch_size =25
epochs = 100
interval = 1

beta = 300
## log setting
log_file = "MinkFPN_GeM"
print_every = 10

save_dir = "/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/Res"
save_prj="experiment_1"
