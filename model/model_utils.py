# Author: Fang Xu, 2022.01.20
# ShenZhen University

from .MinkLoc3d import MinkLoc


def make_model(cfg):

    in_channels = 1
    if 'MinkFPN' in cfg.model_name:
        model = MinkLoc(cfg.model_name, in_channels=in_channels,
                                feature_size = cfg.feature_size,
                                output_dim = cfg.output_dim, planes = cfg.planes,
                                layers = cfg.layers, num_top_down = cfg.num_top_down,
                                conv0_kernel_size = cfg.conv0_kernel_size)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(cfg.model_name))

    return model



