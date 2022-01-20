
from copy import copy
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

transform_depth  = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.PILToTensor()
        ])
depth_path = "/media/fangxu/Disk4T/LQ/data/chess/seq-01/frame-000985.color.png"
img_depth = Image.open( depth_path ).convert('I')
#img_depth = np.array(img_depth)
img_depth= transform_depth(img_depth)

print(img_depth.shape)