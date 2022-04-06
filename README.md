
# Train





#  Update
    2021.8.29: 添加了各种test 函数，修改了数据加载类，修改了深度图像网络，添加了稀疏网络处理模块,模块测试通过。
    2021.9.01: 更新 了训练流程，配置，日志模块，ponit++ backbone 待修改，spbackbone 可以用。

# Usage
    python train.py $config_file_path$

# Question and Description :
**Q1**: 读取点云没有使用降采样，open3d 降采样发现每次得到的点数维度都不一样。既然使用了稀疏处理，就 不用降采样了。，这样就可以解决batch_size 只能为1 的问题，之前好像有一个batch_size 大于1 无法训练，不知道是不是这个原因？（2021.8.29）

**A1**：

**Q2**: 维度关系，不确定现在的维度合并 合不合理？ 深度图像ResNet34 网络采用得到 [Batch, 512, 1, 1], 稀疏网络 得到[10, 15232] 。 合并得到的向量越长，self_attention 是不是处理的内存和参数是不是越来越大？是否足够影响到需要我们在乎这个问题？（2021.8.29）

**A2**：

**D1** : 关于稀疏模块的处理，点云读取得到[batch_size, 307200, 3] , 构建dense data 为 [batch_size, 512,600,3]（2021.8.29）
**D2** : 目前的网络丢弃了 FC层，直接进行合并。（2021.8.29）

# 2. Install

install cndnn

conda install -c conda-forge cudnn

```
export CUDNN_LIBRARY_PATH="/home/fangxu/anaconda3/envs/torch/lib": $CUDNN_LIBRARY_PATH
export CUDNN_INCLUDE_PATH="/home/fangxu/anaconda3/envs/torch/include":$CUDNN_INCLUDE_PATH
```
modify cache file 

https://github.com/traveller59/spconv/issues/277

```
And fixed it by pointing CMake to the appropriate directories in CMakeCache.txt (for me it's in build/temp.linux-x86_64-3.6/CMakeCache.txt). To do this look for the following lines in your CMakeCache.txt and change them to your cuda / cudnn paths, for example:

//Folder containing NVIDIA cuDNN header files
CUDNN_INCLUDE_DIR:FILEPATH=/home/c2/anaconda3/envs/cp2/include

//Path to a file.
CUDNN_INCLUDE_PATH:PATH=/home/c2/anaconda3/envs/cp2/include

//Path to the cudnn library file (e.g., libcudnn.so)
CUDNN_LIBRARY:FILEPATH=/home/c2/anaconda3/envs/cp2/lib/libcudnn.so

//Path to a library.
CUDNN_LIBRARY_PATH:FILEPATH=/home/c2/anaconda3/envs/cp2/lib/libcudnn.so

```

# 3. Note
强制合并代码
git pull --rebase origin main


Resource :
https://github.com/youngguncho/HourglassPose-Pytorch

https://github.com/V-Soboleva/PoseNet

https://github.com/jac99/MinkLoc3D

https://github.com/jac99/MinkLoc3Dv2