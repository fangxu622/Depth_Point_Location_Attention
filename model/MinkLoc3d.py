# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import MinkowskiEngine as ME
import numpy as np
from model.minkfpn import MinkFPN
from model.netvlad import MinkNetVladWrapper
import model.pooling as pooling


class MinkLoc(torch.nn.Module):
    def __init__(self, model, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size):
        super().__init__()
        self.model = model
        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor
        self.backbone = MinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                conv0_kernel_size=conv0_kernel_size, layers=layers, planes=planes)
        self.n_backbone_features = output_dim

        if model == 'MinkFPN_Max':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = pooling.MAC()
        elif model == 'MinkFPN_GeM':
            assert self.feature_size == self.output_dim, 'output_dim must be the same as feature_size'
            self.pooling = pooling.GeM()
        elif model == 'MinkFPN_NetVlad':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=False)
        elif model == 'MinkFPN_NetVlad_CG':
            self.pooling = MinkNetVladWrapper(feature_size=self.feature_size, output_dim=self.output_dim,
                                              cluster_size=64, gating=True)
        else:
            raise NotImplementedError('Model not implemented: {}'.format(model))

    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        x = self.backbone(x)

        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)
        x = self.pooling(x)
        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor
        return x

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print('Backbone parameters: {}'.format(n_params))
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print('Aggregation parameters: {}'.format(n_params))
        if hasattr(self.backbone, 'print_info'):
            self.backbone.print_info()
        if hasattr(self.pooling, 'print_info'):
            self.pooling.print_info()

def get_coords(data):
    coords = []
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != " ":
                coords.append([i, j])
    return np.array(coords)

def data_loader(
    nchannel=3,
    max_label=5,
    is_classification=True,
    seed=-1,
    batch_size=2,
    dtype=torch.float32,
):
    if seed >= 0:
        torch.manual_seed(seed)

    data = ["   X   ", "  X X  ", " XXXXX "]

    # Generate coordinates
    coords = [get_coords(data) for i in range(batch_size)]
    coords = ME.utils.batched_coordinates(coords)

    # features and labels
    N = len(coords)
    feats = torch.arange(N * nchannel).view(N, nchannel).to(dtype)
    label = (torch.rand(batch_size if is_classification else N) * max_label).long()
    return coords, feats, label
