import torch.nn.functional as F
from .convlstmcell import ConvLSTM
from .unet_parts import *
from . import register_model


@register_model("unet_icip")
class UNet(nn.Module):
    def __init__(self, n_channels=2, args=None, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.rank = args.rank
        self.n_layers = args.layers
        self.bilinear = bilinear
        n_resLayers = 11
        res_layers = []
        self.inc = nn.Conv2d(
            n_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='reflect')
        for _ in range(n_resLayers):
            res_layers.append(ResConv(64))
        self.res_layers = nn.Sequential(*res_layers)
        # (input_size, hidden_size, kernel_size)
        self.recurrent = ConvLSTM(64, 64, 3)
        self.out_layer = nn.Conv2d(
            64,
            self.rank *
            self.n_layers *
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='reflect',
            bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, inputs):
        N, _, _, h, w = x.size()
        out = self.inc(x.view(N, -1, h, w))
        out = self.res_layers(out)
        inputs['lf_states'] = self.recurrent(out, inputs['lf_states'])
        lf = self.out_layer(inputs['lf_states'][1])
        lf = lf.view(N, self.n_layers, self.rank, 3, h, w)
        return self.relu(lf)
