import argparse
from utils import add_logging_arguments


def main(args):
    device = torch.device(
        f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    utils.setup_experiment(args)
    utils.init_logging(args)

    # indices of the input stereo frame in the output LF
    left_view_idx = int(args.angular * (args.angular // 2))
    right_view_idx = int(left_view_idx + (args.angular - 1))

    # this is the view indices for the network to predict LF
    lf_view_idx = [left_view_idx, right_view_idx]
    print(f'using stereo view indices as {lf_view_idx}')

    models_list = []
    trainable_params = []
    # ============== Initialize all network models ===================
    # initialize the lf prediction network V
    lf_model = models.build_model(
        args.model,
        n_channels=len(lf_view_idx) * 3,
        args=args).to(device)
    trainable_params.extend(list(lf_model.parameters()))
    models_list.append(lf_model)

    # initialize the optical flow prediction network O
    flow_model = models.build_model(
        args.flow_model,
        n_channels=3,
        args=args).to(device)
    trainable_params.extend(list(flow_model.parameters()))
    models_list.append(flow_model)

    # initialize the disparity map prediction network D
    disp_model = models.build_model(
        args.disp_model,
        n_channels=3,
        args=args).to(device)
    trainable_params.extend(list(disp_model.parameters()))
    models_list.append(disp_model)

    optimizer = None#torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=args.patience, min_lr=1e-6)
    logging.info(
        f"Built {len(models_list)} models consisting of {sum(p.numel() for p in trainable_params):,} parameters")

    # ========== Initialize the low-rank display model ==============
    if args.display == 'multilayer':
        tensor_display = models.multilayer(
            args.angular,
            args.layers,
            args.inph,
            args.inpw,
            args=args).to(device)
    else:
        print('No valid display type chosen')
        print('exiting')
        exit(0)
    logging.info(
        f"Using the {args.display} display with {args.layers} layers and {args.rank} rank")

    state_dict = utils.load_checkpoint(
        args, models_list, optimizer, scheduler)
    global_step = state_dict['last_step']
    start_epoch = state_dict['epoch']

    test_loader = data.build_dataset(
        args.dataset,
        args.data_path,
        args,
        batch_size=args.batch_size,
        num_workers=16)
    # Track moving average of loss values
    test_meters = {name: utils.AverageMeter() for name in (
        ["test_psnr", "test_ssim", "test_loss", "test_lpips"])}
    writer = SummaryWriter(
        log_dir=args.experiment_dir) if not args.no_visual else None
    time_meters = utils.AverageMeter()

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)


    # new save path will be near where the test data is
    # save_path = os.path.join(args.data_path, args.h5_file).split('/')[:-1]
    # save_path = os.path.join(*save_path)
    # expt_dir = args.restore_file.split('/')[-3]
    save_path = args.save_dir#os.path.join('/', save_path, expt_dir)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f'removing the directory tree {save_path}')
    os.makedirs(save_path, exist_ok=True)
    print(save_path)

    for model in models_list:
        model.eval()
    tensor_display.eval()

    def compute_lpips_ssim_psnr(pred_lf, gt_lf, outputs):
        # pred_lf is [N, V, C, h,w]
        N, V, C, H, W = pred_lf.size()
        pred_lf = pred_lf.view(-1, C, H, W)
        gt_lf = gt_lf.view(-1, C, H, W)

        # compute lpips
        pred_lf_norm = 2 * pred_lf - 1.
        gt_lf_norm = 2 * gt_lf - 1.
        lpips_loss = loss_fn_alex(pred_lf_norm, gt_lf_norm).mean()
        outputs['lpips'].append(lpips_loss)

        # compute ssim
        def tensor2np(tensor): return tensor.data.cpu(
        ).numpy().squeeze().transpose(0, 2, 3, 1)
        pred_lf_np = tensor2np(pred_lf)
        gt_lf_np = tensor2np(gt_lf)

        totalssim = 0.
        for k in range(len(pred_lf_np)):
            totalssim += ssim(pred_lf_np[k, ...], gt_lf_np[k, ...],
                              multichannel=True, data_range=1.)
        totalssim /= len(pred_lf_np)
        outputs['ssim'].append(totalssim)

        # compute psnr
        mse = ((pred_lf_np - gt_lf_np)**2).mean()
        psnr = 20 * np.log10(1. / mse)
        outputs['psnr'].append(psnr)

    def run_batch(inputs, outputs):
        inputs['video'] = inputs['video'].to(device)
        inputs['lf_states'] = None
        inputs['prev_idx'] = 0
        inputs['curr_step'] = 0
        outputs['pred_lf'] = []
        outputs['ssim'] = []
        outputs['lpips'] = []
        outputs['psnr'] = []

        for t in range(1, inputs['video'].size(1)):
            # iterate over the t \in T frames
            if t > 0:
                inputs['flow_loss'] = True
                inputs['prev_idx'] = int(t - 1)
            inputs['curr_step'] = t
            run_instance(inputs, outputs)

    def run_instance(inputs, outputs):
        # [N,2,2,3,H,W]: targets
        # [batch, time, view, rgb, height, width]
        targets = inputs['video'][:, inputs['prev_idx']:inputs['curr_step'] + 1, ...]
        curr_gt_lf_frame = targets[:, -1, ...]
        curr_stereo_frame = curr_gt_lf_frame[:, lf_view_idx, ...]
        instance_loss = 0.

        # from the same function return the disparity map
        decomposition = lf_model(curr_stereo_frame, inputs)
        # decomposition is of size [N,layers,rank,3,h,w]
        curr_lf = tensor_display(decomposition)
        curr_lf = curr_lf.clamp(0., 1.)
        outputs['pred_lf'].append(curr_lf)

        # compute the psnr, ssim and lpips values and update
        compute_lpips_ssim_psnr(
            outputs['pred_lf'][-1], curr_gt_lf_frame, outputs)

    # test loader
    for meter in test_meters.values():
        meter.reset()

    test_bar = utils.ProgressBar(test_loader)
    save_every = 1
    metrics_file = open(f'{save_path}/metrics.txt', 'w')
    for sample_id, inputs in enumerate(test_bar):
        ssim_vid = []
        lpips_vid = []
        with torch.no_grad():
            outputs = {}
            run_batch(inputs, outputs)
            # outputs["pred_lf"] will be a sequence of LF frames
            # you just have to save them
            # and also compute psnr; ssim and lpips
            # which you can do in the run_batch fn itself
            # test only with a batch size of 1
            assert inputs['video'].size(0) == 1
            if sample_id % save_every == 0:
                # each video sequence will be saved in a separate directory
                # each frame of the video sequence will be saved in a separate
                # sub-directory
                seq_save_path = f'{save_path}/seq_{sample_id:03d}'
                os.makedirs(f'{save_path}/seq_{sample_id:03d}', exist_ok=True)

                # then save the predicted light field
                for t in range(len(outputs['pred_lf'])):
                    pred_lf_np = outputs['pred_lf'][t].data.cpu(
                    ).numpy().squeeze()
                    pred_lf_np = np.transpose(pred_lf_np, [1, 2, 3, 0])
                    save_lf_path = os.path.join(
                        seq_save_path, f'pred_lf_{sample_id:02d}_{t:02d}.avi')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(
                        save_lf_path, fourcc, 5, (args.inpw, args.inph))
                    for k in range(len(pred_lf_np)):
                        out.write(np.uint8(pred_lf_np[k, ..., ::-1] * 255))

            for metric in ['psnr', 'lpips', 'ssim']:
                mean_val = sum(outputs[metric]) / len(outputs[metric])
                outputs[metric] = mean_val
            test_meters["test_psnr"].update(outputs['psnr'])
            test_meters["test_lpips"].update(outputs['lpips'])
            test_meters["test_ssim"].update(outputs['ssim'])
            metrics_text = f'For seq {sample_id:02d}: PSNR={outputs["psnr"]:0.2f}; SSIM={outputs["ssim"]:0.3f}; LPIPS={outputs["lpips"]:0.3f}\n'
            metrics_file.write(metrics_text)
            print(metrics_text)

    if args.psnr:
        logging.info(
            f"PSNR achieved on test data: {test_meters['test_psnr'].avg:0.3f}")
    metrics_text = f"Dataset average: PSNR={test_meters['test_psnr'].avg:0.2f}; SSIM={test_meters['test_ssim'].avg:0.3f}; LPIPS={test_meters['test_lpips'].avg:0.3f}\n"
    time_text = f'Average time taken is {time_meters.avg}\n'
    metrics_file.write(metrics_text)
    metrics_file.write(time_text)
    print(metrics_text)
    metrics_file.close()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument(
        "--data-path",
        default="/data/prasan/datasets/lfv_testFiles/",
        help="path to data directory")
    parser.add_argument(
        "--h5-file",
        default="lf_data.h5",
        help="path to data directory")
    parser.add_argument(
        "--dataset",
        default="dummy",
        help="train dataset name")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="train batch size")

    # Add model arguments
    parser.add_argument(
        "--model",
        default="unet_lf",
        help="model architecture")
    parser.add_argument(
        "--flow-model",
        default="unet_disp",
        help="depth encoder network")
    parser.add_argument(
        "--disp-model",
        default="unet_disp",
        help="depth decoder network")
    parser.add_argument(
        '--rank',
        type=int,
        default=3,
        help='rank of the light field decomposition')
    parser.add_argument(
        '--layers',
        type=int,
        default=3,
        help='number of layers in the LF display')
    parser.add_argument(
        '--inph',
        type=int,
        default=128,
        help='height of input image')
    parser.add_argument(
        '--inpw',
        type=int,
        default=128,
        help='width of input image')
    parser.add_argument(
        '--angular',
        type=int,
        default=7,
        help='angular resolution of the light field')
    parser.add_argument(
        '--display',
        type=str,
        default="angular",
        choices=(
            'angular',
            'multilayer'),
        help='type of display to use (angular,multilayer)')
    parser.add_argument(
        '--seq-len',
        type=int,
        default=5,
        help='video sequence length')

    parser.add_argument(
        "--gpu",
        default="0",
        help="which gpu to use for training")

    # Add loss parameters
    parser.add_argument(
        "--lambda_sm",
        default=0.01,
        type=float,
        help="how much to weight the TV smoothness loss")
    parser.add_argument(
        "--lambda-temp",
        default=0.01,
        type=float,
        help="how much to weight the TV smoothness loss")
    parser.add_argument(
        "--metric",
        default='l2',
        type=str,
        help="whether to use perceptual or l2 metric")
    parser.add_argument(
        "--psnr",
        action="store_true",
        help="if true,then compute PSNR from GT")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument(
        "--num-epochs",
        default=500,
        type=int,
        help="force stop training at specified epoch")
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=" Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument(
        "--valid-interval",
        default=1,
        type=int,
        help="evaluate every N epochs")
    parser.add_argument(
        "--save-interval",
        default=1,
        type=int,
        help="save a checkpoint every N steps")

    parser.add_argument(
        "--test",
        action="store_true",
        help="if true, then use the RGB test data")
    # Parse twice as model arguments are not known the first time
    # not really parsing twice; Just adding more arguments to the model
    parser = add_logging_arguments(parser)
    # parsing only the known arguments; arguments that are not passed are
    # ignored
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    import os
    import warnings
    warnings.filterwarnings("ignore")
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"set gpu device to {args.gpu}")
    import logging
    import sys
    import torch
    import torchvision
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter
    import imageio
    import pygifsicle
    import lpips
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    import shutil
    import cv2
    import time

    import data
    import models
    import utils

    main(args)
