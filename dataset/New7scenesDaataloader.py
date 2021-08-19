import torch
from PIL import Image
# import torch.utils.data as Dataset
import torch.utils.data as data
import quaternion
import os
import numpy as np
import cv2
import glob, open3d
from torchvision import transforms


class data2d3d_loader(data.Dataset):

    def __init__(self, data_dir ="" , pcd_dir = "/media/fangxu/Disk4T/LQ/pointcloud", scene ="chess",seq_list=[1,2,3,4], voxel_size=0.1, transform_rgb=None, transform_depth = None):
        super(data2d3d_loader, self).__init__()

        # datadir ：
        pcd_dir = os.path.join(pcd_dir,scene)
        imgs_path = []
        depth_path=[]
        pcd_path = []
        labels_path = []
        for i in seq_list:
            pcd_dir_tmp = ""
            if i<10:
                seq_idx = "seq-0{}".format(i)
                #sequenceDir = data_dir + "/seq-0{}/".format(i)
            else:
                seq_idx = "seq-{}".format(i)
            
            sequenceDir = os.path.join(data_dir ,seq_idx)
            pcd_dir_tmp = os.path.join(pcd_dir ,seq_idx)
            poselabelsNames = glob.glob(sequenceDir+"/*.pose.txt")
            poselabelsNames.sort()

            for label in poselabelsNames:
                labels_path.append( label )
                depth_path.append( label.replace("pose.txt","depth.png") )
                imgs_path.append( label.replace("pose.txt","color.png") )
                pcd_path.append( os.path.join(pcd_dir_tmp, label.split("/")[-1].replace("pose.txt","cloud.ply"))  )

        self.voxel_size =voxel_size
        self.data_dir = data_dir
        self.pcd_dir = pcd_dir
        self.imgs_path = imgs_path
        self.depth_path = depth_path
        self.pcd_path = pcd_path
        self.transform_depth = transform_depth
        self.labels_path = labels_path

        if transform_depth == None:
            self.transform_depth = transforms.Compose([
                               transforms.ToTensor(),
                               # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                             ])
        else:
            self.transform_depth = transform_depth

    def __getitem__(self, index):

        #img_depth = cv2.imread( self.depthimgs[index] , 2 ) # cv2.IMREAD_ANYDEPTH
        img_depth = Image.open( self.depth_path[index] ).convert('I')
        img_depth= self.transform_depth(img_depth)
        img_depth = img_depth.type(torch.FloatTensor)/1000.0

        pcd_data =  open3d.io.read_point_cloud( self.pcd_path[index] )
        downpcd = pcd_data.voxel_down_sample(voxel_size= self.voxel_size)
        downpcd_arr = np.asarray(downpcd.points)
        downpcd_arr = downpcd_arr.astype(np.float)
        # print(downpcd_arr.shape)
        pose = np.loadtxt( self.labels_path[index] )
        q =  quaternion.from_rotation_matrix(pose[:3,:3] )
        t = pose[:3,3]
        q_arr = quaternion.as_float_array(q)#[np.newaxis,:]

        result = ( torch.tensor(img_depth).to(torch.float32) , torch.tensor(downpcd_arr).to(torch.float32), torch.tensor(t).to(torch.float32), torch.tensor(q_arr).to(torch.float32) )

        return  result

    def __len__(self):
        return len(self.labels_path)


