
from cgi import test
from turtle import clear
import os, torch, math
from torch import nn
from platform import machine
import open3d 
import open3d as o3d
import numpy as np
import sys, logging, time
sys.path.append("/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention")
from tqdm import tqdm
from dataset import make_dataloaders,make_datasets
from model import Fuse_PPNet, Pose_Depth_Net, standard_pose_loss, Pose_Pcd_Net,make_model
from mmcv import Config
import MinkowskiEngine as ME

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.get_device_name(0))

torch.cuda.manual_seed(1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_o3d():
    print("Load a ply point cloud, print it, and render it")
    #ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud("/home/fangxu/Downloads/fragment.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

def test_pcd():
    pcd_path = "/media/fangxu/Disk4T/LQ/pointcloud/chess/seq-01_local/frame-000023.cloud.ply"
    #pcd_path = "/home/fangxu/Downloads/fragment.ply"
    pcd_data =  o3d.io.read_point_cloud( pcd_path )
    #pcd_data = pcd_data.voxel_down_sample(voxel_size=0.04)
    pcd_data = pcd_data.uniform_down_sample(30)
    pcd_data = np.asarray(pcd_data.points)

    print(pcd_data.shape)
    #print(pcd_data)

def get_coords(data):
    coords = []
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != " ":
                coords.append([i, j,j+1])
    return np.array(coords)

def test_data_loader(
    nchannel=3,
    max_label=5,
    is_classification=True,
    seed=-1,
    batch_size=2,
    dtype=torch.float32):

    if seed >= 0:
        torch.manual_seed(seed)

    data = ["   X   ", "  X X  ", " XXXXX "]

    # Generate coordinates
    coords1 = [get_coords(data) for i in range(batch_size)]

    print(coords1[0])

    coords = ME.utils.batched_coordinates(coords1)

    # features and labels
    N = len(coords)
    feats = torch.arange(N * nchannel).view(N, nchannel).to(dtype)
    label = (torch.rand(batch_size if is_classification else N) * max_label).long()
    print(feats)
    print(coords)
    print(label)
    return coords, feats, label

def test_dataloader():
    config_path = '/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/config/conf_fuse.py'
    assert os.path.exists(config_path)==True
    cfg = Config.fromfile(config_path)

    dataset3d = make_datasets(cfg)
    data_list = dataset3d['train']
    clouds=[]
    for i in range(5):
        clouds.append( data_list[i][1] )
    
    batch_pcd = torch.stack(clouds, dim=0)
    print(batch_pcd.shape)
    print(batch_pcd[ 1, 14:18, :])

    coords = [ ME.utils.sparse_quantize(coordinates=batch_pcd[0], quantization_size=cfg.mink_quantization_size) for e in batch_pcd ]
    coords = ME.utils.batched_coordinates(coords)
    print(coords.shape)
    print(coords[-8:-4,:])

def test_dataloader2():
    config_path = '/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/config/conf_fuse.py'
    assert os.path.exists(config_path)==True
    cfg = Config.fromfile(config_path)

    train_loader , test_loader = make_dataloaders(cfg)
    i, (depth_base , pcd_base,  base_t,base_q) = enumerate(train_loader).__next__()

    print( depth_base.shape)
    print( pcd_base.keys() )
    print( pcd_base['coords'].shape )
    print( pcd_base['coords'][1:8,:])
    print( pcd_base['features'].shape )
    print( pcd_base['features'][1:8,:] )

def test_backbone():
    config_path = '/media/fangxu/Disk4T/fangxuPrj/Depth_Point_Location_Attention/config/conf_pcd.py'
    assert os.path.exists(config_path)==True
    cfg = Config.fromfile(config_path)

    model = make_model(cfg).to(device)
    # Pcd_Net.train()
    #model =  Pose_Pcd_Net(cfg)
    model.to(device)

    train_loader , test_loader = make_dataloaders(cfg)
    i, (depth_base , pcd_base,  base_t,base_q) = enumerate(train_loader).__next__()

    pcd_input = {e: pcd_base[e].to(device) for e in pcd_base}
    t_gt, q_gt = base_t.to(device) , base_q.to(device)

    print(t_gt.shape)

    pred = model(pcd_input)

    print(pred.shape)


if __name__=="__main__":
    print("x")
    #test_o3d()
    test_pcd()
    #test_backbone()
    #test_data_loader()
    #test_dataloader()
