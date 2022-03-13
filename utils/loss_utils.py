import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.autograd import Variable
from .msssim import *


# def gradxy2(tensor):
#     grad = F.conv2d(tensor,grad_var2.to(tensor.device),padding=1,stride=1)
#     return grad

# def gradxy(tensor):
#     grad = F.conv2d(tensor,grad_var.to(tensor.device),padding=1,stride=1)
#     return grad

# def save_gif(arr, path, fps = 2,normalize=True):
#     # (9,H,W)
#     frames = []
#     dur = 1000./fps
#     for sub_frame in range(arr.shape[0]):
#         img = arr[sub_frame,...]
#         # print(np.amax(img), np.amin(img))
#         if normalize:
#             img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
#         img = img.clip(0,1)
#         img = Image.fromarray((img*255.).astype('uint8'))
#         # img = Image.fromarray(img)
#         frames.append(img)
#     frame1 = frames[0]
#     frame1.save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0)
#     return

# def unfold(tensor,ss):
#     # tensor is of size [N,3,64,375,540]
#     # output should be like [N',3,64,ss,ss]
#     out_tensor = []
#     N,_,a,h,w = tensor.size()
#     for k in range(tensor.size(1)):
#         ct = tensor[:,k,...]
#         ct_unfold = ct.unfold(2,ss,120).unfold(3,ss,128)
#         ct_unfold = ct_unfold.permute(0,2,3,1,4,5)
#         ct_unfold = ct_unfold.reshape(-1,1,a,ss,ss)
#         out_tensor.append(ct_unfold)
#     return torch.cat(out_tensor,1)

# def make_grid_sinusoidal(h,n_scales=4):
#     x = np.linspace(0.,h,h)
#     y = np.linspace(0.,h,h)
#     xv,yv = np.meshgrid(x,y)
#     default_flow = np.zeros((1,2,h,h))
#     default_flow[0,0,...] = xv
#     default_flow[0,1,...] = yv
#     x_grid = np.zeros((1,2*n_scales,h,h))
#     y_grid = np.zeros((1,2*n_scales,h,h))
#     for k in range(n_scales):
#         coeff_x = 2*3.14*xv/(2**k)
#         x_grid[0,2*k,...] = np.sin(coeff_x)
#         x_grid[0,2*k+1,...] = np.cos(coeff_x)
#         coeff_y = 2*3.14*yv/(2**k)
#         y_grid[0,2*k,...] = np.sin(coeff_y)
#         y_grid[0,2*k+1,...] = np.cos(coeff_y)
#     grid_sin = np.concatenate([x_grid,y_grid],1)
#     return torch.FloatTensor(grid_sin)

# def normalize(lf):
#     # lf_min = lf.min(1,True)[0].min(2,True)[0].min(3,True)[0]
#     lf_max = lf.max(1,True)[0].max(2,True)[0].max(3,True)[0]
#     # lf = (lf - lf_min)/(lf_max - lf_min)
#     return lf/lf_max

# def smoothness_loss(img):
#     img_grad = gradxy(img)
#     img_grad = img_grad.pow(2)
#     loss = img_grad.mean()
#     return loss

# def weighted_l2(pred,target):
#     grad_weight = gradxy(target)
#     grad_weight = grad_weight.abs() + 1.
#     recon_loss = ((target-pred)*grad_weight).pow(2).mean()
#     return recon_loss

class sm_loss(object):
    """docstring for depth_sm_loss
    smoothness loss from https://arxiv.org/pdf/2004.11364.pdf
    Eq. 10 and 11; page 4
    """

    def __init__(self, args=None):
        super(sm_loss, self).__init__()
        device = torch.device(
            f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        sobel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = sobel_x.T
        self.gradx = torch.FloatTensor(sobel_x[None, None, ...]).to(device)
        self.grady = torch.FloatTensor(sobel_y[None, None, ...]).to(device)

        # parameters as defined in https://arxiv.org/pdf/2004.11364.pdf
        self.emin = 0.1
        self.gmin = 0.05

    def gradxy(self, tensor):
        gradx = F.conv2d(tensor, self.gradx, padding=1, stride=1)
        grady = F.conv2d(tensor, self.grady, padding=1, stride=1)
        grad = gradx.abs() + grady.abs()
        return grad

    def edge_mask(self, Is):
        '''
        Is is the gt image of size [N,C,H,W]
        '''
        Is = Is.mean(1, keepdim=True)
        Gs = self.gradxy(Is)
        max_Gs = torch.max(Gs) + 1e-3
        Es = torch.min(Gs / (self.emin * max_Gs), torch.ones_like(Gs))
        return Es

    def compute_loss(self, D, I):
        '''
        given single depth and single I
        compute the loss
        '''
        Es = self.edge_mask(I)
        G_D = self.gradxy(D)
        loss = torch.max(G_D - self.gmin, torch.zeros_like(G_D)) * (1. - Es)
        return loss.mean()

    def edge_aware_loss(self, pred_depth, gt_stereo):
        '''
        pred_depth: [N,2,H,W]
        gt_stereo: [N,2*C,H,W]
        '''
        C = int(gt_stereo.size(2) // 2)
        loss = self.compute_loss(pred_depth[0], gt_stereo[:, 0, ...]) + \
            self.compute_loss(pred_depth[1], gt_stereo[:, 1, ...])
        return loss

    def image_sm(self, pred_lf):
        '''
        pred_depth: [N,25,H,W]
        '''
        C = int(pred_lf.size(1))
        h, w = pred_lf.size(-2), pred_lf.size(-1)
        pred_lf = pred_lf.view(-1, 1, h, w)
        grads = self.gradxy(pred_lf)

        return grads.mean()

    def flow_sm(self, pred_flow):
        '''
        pred_depth: [N,25,H,W]
        '''
        loss = 0.
        for flow in pred_flow:
            h, w = flow.size(-2), flow.size(-1)
            flow = flow.view(-1, 1, h, w)
            grads = self.gradxy(flow)
            loss += grads.mean()
        return loss

# # compute ssim metric for test and not loss
# def compute_ssim(pred,gt):
#     # target has single channel
#     # while pred has angular**2 channels
#     ssim_fn = SSIM(window_size=11)
#     ssim_loss = 0.
#     for k in range(pred.size(1)):
#         ssim = ssim_fn(pred[:,k:k+1,...],gt[:,k:k+1,...])
#         ssim_loss += ssim
#     return ssim_loss/(pred.size(1))


class temporal_criterion(object):
    """docstring for temporal_criterion"""

    def __init__(self, args=None):
        super(temporal_criterion, self).__init__()
        # grid of multipliers
        self.device = torch.device(
            f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        self.angular = args.angular

        self.default_flow = self.get_default_flow(
            args.inph, args.inpw).to(self.device)

    def get_default_flow(self, h, w):
        x = np.linspace(-1., 1., w)
        y = np.linspace(-1., 1., h)
        xv, yv = np.meshgrid(x, y)
        default_flow = np.zeros((h, w, 2))
        default_flow[..., 0] = xv
        default_flow[..., 1] = yv
        default_flow = default_flow[None, ...]
        # default_flow = np.tile(default_flow,[1,2,1,1,1])
        default_flow = torch.FloatTensor(default_flow)
        return default_flow

    def consistency(self, time_stereo, pred_flow, detach=False):
        # first, need to transpose the flow
        # then add the default flow to it
        # then warp the temporal frames from one to another
        loss = 0.
        time_left = [time_stereo[0][:, 0, ...],
                     time_stereo[1][:, 0, ...]]  # [prev, current]
        time_right = [time_stereo[0][:, 1, ...], time_stereo[1][:, 1, ...]]
        # time_series = [time_left, time_right]
        # for k in range(len(time_series)):
        for flow, time_series in zip(pred_flow, [time_left, time_right]):
            flow = flow.permute(0, 2, 3, 1).contiguous()
            if detach:
                flow = flow.detach()
            flow = flow + self.default_flow
            warped_img = F.grid_sample(
                time_series[1],
                flow,
                mode='bilinear',
                align_corners=True)
            loss += (warped_img - time_series[0]).abs().mean()
        return loss


class loss_geo_stereo_disp(object):

    def __init__(self, args=None):
        super(loss_geo_stereo_disp, self).__init__()

        # to compute gradients
        self.depth_sm = sm_loss(args=args)
        # grid of multipliers
        angular = args.angular
        self.device = torch.device(
            f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        self.angular = args.angular

        # grid for horizontal extreme views
        x = np.arange(0, angular)
        y = np.arange(-angular // 2 + 1, angular // 2 + 1)
        X, Y = np.meshgrid(x, y)

        mul_grid = np.stack([X, Y], 2)
        mul_grid = mul_grid * 1.0  # /args.angular
        mul_grid = np.reshape(mul_grid, [angular**2, 1, 1, 2])
        self.gridTensor = torch.FloatTensor(
            mul_grid[None, ...]).to(self.device)

        x = x - (angular - 1)
        X, Y = np.meshgrid(-1. * x, -1. * y)
        mul_grid = np.stack([X, Y], 2)
        # mul_grid = mul_grid*1.0#/args.angular
        mul_grid = np.reshape(mul_grid, [angular**2, 1, 1, 2])
        self.gridTensor2 = torch.FloatTensor(
            mul_grid[None, ...]).to(self.device)

        self.default_flow = self.get_default_flow(
            args.inph, args.inpw).to(self.device)

    def normalized_l1(self, pred, target, reduce_mean=True):
        '''
        loss will be normalized by the values in target
        '''
        # grad_weight = (5.*self.depth_sm.gradxy(target)) + 1.
        l1_loss = ((pred - target).abs())  # *grad_weight
        # compute ssim
        # lpips = self.compute_lpips(pred,target)
        # loss = 0.8*ssim + 0.2*l1_loss
        loss = l1_loss
        if reduce_mean:
            return loss.mean()
        else:
            return loss

    def get_default_flow(self, h, w):
        x = np.linspace(-1., 1., w)
        y = np.linspace(-1., 1., h)
        xv, yv = np.meshgrid(x, y)
        default_flow = np.zeros((h, w, 2))
        default_flow[..., 0] = xv
        default_flow[..., 1] = yv
        default_flow = default_flow[None, None, ...]
        default_flow = np.tile(default_flow, [1, self.angular**2, 1, 1, 1])
        default_flow = torch.FloatTensor(default_flow)
        return default_flow

    def consistency(self, lf, gt_stereo_frame, pred_depth):

        left_idx = int(self.angular * (self.angular // 2))
        right_idx = int(left_idx + (self.angular - 1))

        # get the disparity for the left stereo view
        depth = pred_depth[0]  # [:,:1,...]
        N, _, H, W = depth.size()
        # get the total nr of lf views
        V = lf.size(1)
        C = lf.size(2)  # no of channels: 3 for RGB
        depth = depth.unsqueeze(4)
        # repeat the disparity in both x and y directions
        # only need to repeat by V because the shift
        # remains the same for all 3 RGB channels
        depth = depth.repeat(1, V, 1, 1, 2)
        diff = 0.
        # multiply with the grid so that the disparities are appropriately scaled
        # to warp each LF view
        depthx = depth * self.gridTensor

        # add the default flow to make it compatible with pytorch warping fn.
        depthx = depthx + self.default_flow
        N, _, H, W, _ = depthx.size()

        # now select the scaled disparity map that
        # warps the right view to the left view
        right_disp = depthx[:, right_idx, ...]
        depthx = depthx.reshape(N * V, H, W, 2)
        lf = lf.reshape(N * V, C, H, W)

        # append the gt right view to the predicted light field
        # but the predicted gt right view is of 3 channels
        lf_plusright = torch.cat([lf, gt_stereo_frame[:, 1, ...]], 0)

        # append the scaled disparity that warps the right stereo view
        # and the right disparity to the whole disparity field
        # right_disp is appended twice because we need to warp both gt right view
        # and the predicted right view disparity
        depthx = torch.cat([depthx, right_disp], 0)

        warped_lf = F.grid_sample(
            lf_plusright,
            depthx,
            padding_mode='border',
            mode='bilinear',
            align_corners=True)

        # compute losses
        pred_warped_lf = warped_lf[:N * V, ...].reshape(N, V, C, H, W)
        target1_loss = self.normalized_l1(
            pred_warped_lf, gt_stereo_frame[:, 0, ...].unsqueeze(1), reduce_mean=False)
        # lpips1 = self.compute_lpips(pred_warped_lf , gt_stereo_frame[:,:1,:,:])

        # compute losses for the ground truth stereo view
        right_warped = warped_lf[N * V:N * V + N, ...]
        # loss for warping right image (source) to left image (target)
        right_warp_loss = self.normalized_l1(
            right_warped, gt_stereo_frame[:, 0, ...])

        # repeat the above for the right frame
        depth = pred_depth[1]  # [:,1:,...]
        depth = depth.unsqueeze(4)
        # depth = depth.expand(N,C,H,W,2)
        depth = depth.repeat(1, V, 1, 1, 2)
        depthx = depth * self.gridTensor2
        depthx = depthx + self.default_flow
        N, _, H, W, _ = depthx.size()
        left_disp = depthx[:, left_idx, ...]
        depthx = depthx.reshape(N * V, H, W, 2)
        lf_plusleft = torch.cat([lf, gt_stereo_frame[:, 0, ...]], 0)
        depthx = torch.cat([depthx, left_disp], 0)

        warped_lf = F.grid_sample(
            lf_plusleft,
            depthx,
            padding_mode='border',
            mode='bilinear',
            align_corners=True)
        pred_warped_lf = warped_lf[:N * V, ...].reshape(N, V, C, H, W)
        target2_loss = self.normalized_l1(
            pred_warped_lf, gt_stereo_frame[:, 1, ...].unsqueeze(1), reduce_mean=False)

        left_warped_lf = warped_lf[N * V:N * V + N, ...]
        left_warp_loss = self.normalized_l1(
            left_warped_lf, gt_stereo_frame[:, 1, ...])

        # total loss
        # from eq(4) of https://arxiv.org/pdf/1806.01260.pdf
        # min_reproj_loss = torch.min(target1_loss,target2_loss).mean()# if
        # e>20 else 0
        min_reproj_loss = (target1_loss + target2_loss).mean()
        total_loss = min_reproj_loss + \
            left_warp_loss + \
            right_warp_loss
        return total_loss
