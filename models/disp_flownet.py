""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .convlstmcell import ConvLSTM
from .unet_parts import *
from . import register_model
# https://github.com/ClementPinard/Pytorch-Correlation-extension
from spatial_correlation_sampler import SpatialCorrelationSampler
# from .corr1d import Corr # cost volume for disparity

'''
Documentation for SpatialCorrelationSampler
    kernel is the radius of area used for computing correlation. It is very similar to convolution kernel, since it's a weighted sum of the neighbourhood of a particular point, with the difference that here the sum is weighted by the corresponding neighbourhood of the second input.

    patch is the different translations we test between input1 and input2. Regular correlation is done with a translation of (0,0), and then we test different translations. for a particular translation u,v, we compute correlIation between input1[:, u:, v:] and input2[:, :-u, :-v] .

    dilation is the same as dilation in regular convolution, meaning that e.g. for a 3x3 correlation with dilation 2, you actually have 5x5 "a trous" correlation. This is not dependent to different translations tested in patch

    the corresponding dilation paremeter for patch is dilation_patch. Instead of testing every possible integer translation in [-patch_size/2, +patch_size/2] , we multiply every translation tested by dilation_patch.
    (Meaning we test as many different translation as with a dilation_patch of 1, but the translation are enhanced
'''


class BaseNetC(nn.Module):
    def __init__(self, n_channels=2, args=None, bilinear=True, mode='disp'):
        super(BaseNetC, self).__init__()
        self.n_channels = n_channels
        self.rank = args.rank
        self.n_layers = args.layers
        self.bilinear = bilinear

        inc = [64, 64, 128, 128, 256]

        # self.inc = DoubleConv(n_channels, inc[0])
        # self.down1 = DownRes(inc[0], inc[1])
        # self.conv_redir = DoubleConv(inc[1],inc[1]//2)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.conv_redir = DoubleConv(256, 256 // 2)

        print(f'using mode {mode}')
        if mode == 'disp':
            # previous value 5
            ps = (1, 11)
            self.cv = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=ps,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=2)
        else:
            # previous value 11
            ps = (11, 11)
            self.cv = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=ps,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=2)
        self.cv_channels = ps[0] * ps[1]

        # self.down2 = DownRes(self.cv_channels + inc[1]//2, inc[2])
        # self.down3 = DownRes(inc[2], inc[3])
        # factor = 2 if bilinear else 1
        # self.down4 = DownRes(inc[3], inc[4] // factor)

        # self.recurrent = ConvLSTM(self.cv_channels + inc[1]//2,
        # self.cv_channels + inc[1]//2, 3) #(input_size, hidden_size,
        # kernel_size)

        # self.up1 = UpRes(inc[4]//factor + inc[3], inc[3], bilinear)
        # self.up2 = UpRes(inc[3] + inc[2]        , inc[2], bilinear)
        # self.up3 = UpRes(inc[2] + inc[1]        , inc[1], bilinear)
        # self.up4 = UpRes(inc[1] + inc[0]        , inc[0], bilinear)
        # if mode == 'disp':
        #     self.outc = OutConv(inc[0], 1)
        # else:
        #     self.outc = OutConv(inc[0], 2)

        self.down3 = Down(self.cv_channels + 256 // 2, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        # (input_size, hidden_size, kernel_size)
        self.recurrent = ConvLSTM(512 // factor, 512 // factor, 3)

        self.up1 = Up(512, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        if mode == 'disp':
            self.outc = OutConv(64, 1)
        else:
            self.outc = OutConv(64, 2)

    def one_branch(self, img0, img1, prev_states, x3, x2, x1):
        cv = self.cv(img0, img1)
        N, _, _, h, w = cv.size()
        cv = cv.view(N, self.cv_channels, h, w)

        # x3 = self.down2(cv)
        x3_redir = self.conv_redir(x3)
        cv = torch.cat([cv, x3_redir], 1)
        x4 = self.down3(cv)
        x5 = self.down4(x4)
        states = self.recurrent(x5, prev_states)
        x = self.up1(states[1], x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        pred = self.outc(x)
        return pred, states


@register_model("flownetC")
class FlowNet(BaseNetC):
    """docstring for FlowNet"""

    def __init__(self, n_channels=2, args=None, bilinear=True):
        super(FlowNet, self).__init__(n_channels, args, bilinear, 'flow')

    def forward(self, x, inputs):
        # x will be of size [N,T=2,2,H,W]
        # for each view (axis 2) I need to predict the flow
        # time sequence of the left view
        # left_imgs = torch.chunk(x[:,:,0,...],2,dim=1)
        # imgs[0] will be [N,T=2,H,W]
        left_imgs = x[:, :, 0, ...]
        # N,_,_,h,w = left_imgs[0].size()
        x1_t1 = self.inc(left_imgs[:, 0, ...])
        x2_t1 = self.down1(x1_t1)
        x3_t1 = self.down2(x2_t1)

        x1_t = self.inc(left_imgs[:, 1, ...])
        x2_t = self.down1(x1_t)
        x3_t = self.down2(x2_t)

        flow_left, states_left = self.one_branch(
            x3_t1, x3_t, inputs['states_flow'][0], x3_t1, x2_t1, x1_t1)

        right_imgs = x[:, :, 1, ...]
        # imgs[0] will be the timestep to predict flow
        x1_t1 = self.inc(right_imgs[:, 0, ...])
        x2_t1 = self.down1(x1_t1)
        x3_t1 = self.down2(x2_t1)

        x1_t = self.inc(right_imgs[:, 1, ...])
        x2_t = self.down1(x1_t)
        x3_t = self.down2(x2_t)

        flow_right, states_right = self.one_branch(
            x3_t1, x3_t, inputs['states_flow'][1], x3_t1, x2_t1, x1_t1)
        inputs['states_flow'] = [states_left, states_right]
        return [flow_left, flow_right]


@register_model("dispnetC")
class DispNet(BaseNetC):
    """docstring for FlowNet"""

    def __init__(self, n_channels=3, args=None, bilinear=True):
        super(DispNet, self).__init__(n_channels, args, bilinear, 'disp')

    def forward(self, x, inputs):
        # x will be of size [N,2,3,H,W]
        # for each view (axis 1) I need to predict the disparity
        imgs = torch.chunk(x, 2, dim=1)

        x1_l = self.inc(x[:, 0, ...])
        x2_l = self.down1(x1_l)
        x3_l = self.down2(x2_l)

        x1_r = self.inc(x[:, 1, ...])
        x2_r = self.down1(x1_r)
        x3_r = self.down2(x2_r)
        pred_left, states_left = self.one_branch(
            x3_l, x3_r, inputs['states_disp'][0], x3_l, x2_l, x1_l)
        pred_right, states_right = self.one_branch(
            x3_r, x3_l, inputs['states_disp'][1], x3_r, x2_r, x1_r)
        inputs['states_disp'] = [states_left, states_right]
        return [10. * pred_left, 10. * pred_right]
