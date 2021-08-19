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

git pull --rebase origin main