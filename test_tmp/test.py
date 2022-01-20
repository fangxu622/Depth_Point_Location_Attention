
import os
from mmcv import Config
import sys
sys.path.append("/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention")
from dataset import make_dataloaders, make_datasets
from model import make_model

def test_data2():
    batch_size =5
    config_path = '/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/config/conf_1.py'
    assert os.path.exists(config_path)==True
    cfg = Config.fromfile(config_path)
    dataset7scene = make_datasets(cfg)
    print(len(dataset7scene['train']))

    train_loader, test_loader = make_dataloaders(cfg)
    print(len(train_loader))

    model = make_model(cfg)

    for i, (img_base , pcd_base , base_t,base_q) in enumerate(train_loader):

        if i  == 2:
            print('Batch {} of {}'.format(i, len(train_loader)))
            # imgs_base = Variable(img_base.type(dtype))
            print(img_base.shape,  pcd_base['coords'].shape,  pcd_base['features'].shape, base_t.shape, base_q.shape) # torch.Size([5, 1, 224, 224]) torch.Size([5, 307200, 3]) torch.Size([5, 3]) torch.Size([5, 4])
            
            x = model(pcd_base)
            #x = ME.SparseTensor( pcd_base['features'], coordinates= pcd_base['coords'])
            print(x.shape)
            break


# if __name__=="__main__":
#     print("x")
test_data2()