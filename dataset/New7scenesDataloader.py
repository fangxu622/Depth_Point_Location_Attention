from ast import ExtSlice
from fnmatch import translate
import torch
from PIL import Image
# import torch.utils.data as Dataset
from torch.utils.data import Dataset
import quaternion
import os
import numpy as np
import cv2
import glob, open3d
from torchvision import transforms
import MinkowskiEngine as ME
from scipy.linalg import expm, norm
import random,math


class data2d3d_loader(Dataset):

    def __init__(self, data_dir ="" , pcd_dir = "/media/fangxu/Disk4T/LQ/pointcloud", scene ="chess",seq_list=[1,2,3,4], voxel_size=0.05, transform_rgb=None, transform_depth = None):
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
        #downpcd = pcd_data.voxel_down_sample(voxel_size= self.voxel_size)
        downpcd_arr = np.asarray(pcd_data.points)
        #downpcd_arr = downpcd_arr.astype(np.float)
        # print(downpcd_arr.shape)
        pose = np.loadtxt( self.labels_path[index] )
        q =  quaternion.from_rotation_matrix(pose[:3,:3] )
        t = pose[:3,3]
        q_arr = quaternion.as_float_array(q)#[np.newaxis,:]

        result = ( img_depth.to(torch.float32) , torch.from_numpy(downpcd_arr).to(torch.float32), torch.from_numpy(t).to(torch.float32), torch.from_numpy(q_arr).to(torch.float32) )

        return  result

    def __len__(self):
        return len(self.labels_path)


class New7Scene_dataset(Dataset):

    def __init__(self, data_dir ="" , pcd_dir = "/media/fangxu/Disk4T/LQ/pointcloud", scene ="chess",seq_list=[1,2,3,4], voxel_size=0.05, transform_rgb=None, transform_depth = None, transform_pcd= None, set_transform = None):
        super(New7Scene_dataset, self).__init__()

        # datadir ：
        pcd_dir = os.path.join(pcd_dir,scene)
        data_dir = os.path.join(data_dir,scene)
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
        self.transform_pcd = transform_pcd
        self.set_transform = set_transform
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
        img_depth = img_depth.type(torch.float)/1000.0

        pcd_data =  open3d.io.read_point_cloud( self.pcd_path[index] )
        pcd_tensor = torch.from_numpy(np.asarray(pcd_data.points)).to(torch.float)
        if self.transform_pcd is not None:
            pcd_tensor = self.transform_pcd(pcd_tensor)

        pose = np.loadtxt( self.labels_path[index] )
        q =  quaternion.from_rotation_matrix(pose[:3,:3] )
        t = pose[:3,3]
        q_arr = quaternion.as_float_array(q)#[np.newaxis,:]

        result = ( img_depth, 
                    pcd_tensor.to(torch.float), 
                    torch.from_numpy(t).to(torch.float), 
                    torch.from_numpy(q_arr).to(torch.float) )

        return  result

    def __len__(self):
        return len(self.labels_path)


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


class TrainTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class TrainSetTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        self.transform = None
        t = [RandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1])),
             RandomFlip([0.25, 0.25, 0.])]
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 < sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta      # Rotation around axis
        self.max_theta2 = max_theta2    # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.max_theta2 is None:
            coords = coords @ R
        else:
            R_n = self._M(np.random.rand(3) - 0.5, (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5))
            coords = coords @ R @ R_n

        return coords


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64 )

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n*r), replace=False)   # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)     # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords

