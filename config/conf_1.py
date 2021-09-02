
from typing import Optional
import torch

#cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


## datasetting
data_dir = "/media/fangxu/Disk4T/LQ/data/"
scene = "chess" #optional "chess", 



## model selection
backbone = "spconvnet" # option : poinit++ , spconvnet

## optimazation setting
learning_rate = 1e-4
batch_size =4
epochs = 500

## log setting
log_file = ""
print_every = 32

save_dir = "/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/Res"
save_prj="experiment_1"