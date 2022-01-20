
import os, torch
from mmcv import Config
import sys
sys.path.append("/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention")
from dataset import make_dataloaders, make_datasets
from model import make_model, Fuse_PPNet,AttentionBlock, Depth_Net


def test_data2():
    batch_size =5
    config_path = '/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/config/conf_1.py'
    assert os.path.exists(config_path)==True
    cfg = Config.fromfile(config_path)
    dataset7scene = make_datasets(cfg)
    print(len(dataset7scene['train']))

    train_loader, test_loader = make_dataloaders(cfg)
    print(len(train_loader))

    model = Fuse_PPNet(cfg=cfg)

    for i, (img_base , pcd_base , base_t,base_q) in enumerate(train_loader):

        if i  == 2:
            print('Batch {} of {}'.format(i, len(train_loader)))
            # imgs_base = Variable(img_base.type(dtype))
            print(img_base.shape,  pcd_base['coords'].shape,  pcd_base['features'].shape, base_t.shape, base_q.shape) # torch.Size([5, 1, 224, 224]) torch.Size([5, 307200, 3]) torch.Size([5, 3]) torch.Size([5, 4])
            
            x = model(img_base,pcd_base)
            #x = ME.SparseTensor( pcd_base['features'], coordinates= pcd_base['coords'])
            print(x[0].shape,x[1].shape)
            break


def test_depth():
    print("x")
    x = torch.randn(10, 1, 224 , 224).cuda()
    D_Net = Depth_Net().cuda()
    x = D_Net(x)
    print(x.shape) #torch.Size([10, 512, 1, 1])


def test_atten():
    x = torch.randn(10, 12).cuda()
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
    test_data2()