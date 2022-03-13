import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def get_default_flow(b, h, w):
    x = np.linspace(-1., 1., w)
    y = np.linspace(-1., 1., h)
    xv, yv = np.meshgrid(x, y)
    default_flow = np.zeros((h, w, 2))
    default_flow[..., 0] = xv
    default_flow[..., 1] = yv
    return default_flow


class multilayer(nn.Module):
    def __init__(self, angular, n_layers, h, w, reduce_mean=True, args=None):
        super(multilayer, self).__init__()
        # define the filters
        self.reduce_mean = reduce_mean
        self.device = torch.device(
            f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        disp_level_h = (-1.0 / h)
        disp_level_w = (-1.0 / w)
        self.a = angular
        disp_layers = np.zeros((angular**2, h, w, 2))
        default_flow = get_default_flow(1, h, w)
        a = 0
        for k in range(-angular // 2 + 1, angular // 2 + 1):
            for l in range(-angular // 2 + 1, angular // 2 + 1):
                disp_layers[a, :, :, 0] = disp_level_w * l
                disp_layers[a, :, :, 1] = disp_level_h * k
                a += 1
        layers = []
        for c in range(-n_layers // 2 + 1, n_layers // 2 + 1):
            layers.append(disp_layers * c)
        layers = np.stack(layers, 0)
        layers += default_flow[None, None, ...]
        self.filters = torch.FloatTensor(layers).to(self.device)

    def cust_expand(self, l):
        N, _, h, w = l.size()
        # l = F.relu(l)
        l = l.expand(N, self.a**2, h, w)
        l = l.unsqueeze(2)
        return l

    def forward(self, layers):
        N, num_layers, rank, c, h, w = layers.size()  # c is the RGB channel
        lf = []
        filters = self.filters.unsqueeze(0).expand(
            N, num_layers, self.a**2, h, w, 2)
        # layers is of shape [n_layers,T,h,w]
        for a in range(self.a**2):
            layers_shift_prod = torch.ones(N, rank * c, h, w).to(self.device)
            for l in range(num_layers):
                layer = layers[:, l, ...].view(N, rank * c, h, w)
                layers_shift = F.grid_sample(
                    layer, filters[:, l, a, ...], padding_mode='border', mode='bilinear', align_corners=True)
                layers_shift_prod = layers_shift_prod * layers_shift
            layers_shift_prod = layers_shift_prod.view(N, rank, c, h, w)
            if self.reduce_mean:
                lf.append(layers_shift_prod.mean(1, keepdim=True))
            else:
                lf.append(layers_shift_prod)
        if self.reduce_mean:
            lf = torch.cat(lf, 1)
            return lf
        else:
            lf = torch.stack(lf, 2)
            return lf
