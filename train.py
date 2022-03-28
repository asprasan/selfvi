import argparse
from utils import add_logging_arguments


def main(args):
    '''
    Official implementation of
    Shedligeri, Prasan, et al. 
    "SeLFVi: Self-Supervised Light-Field Video Reconstruction From Stereo Video." 
    Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    link: https://openaccess.thecvf.com/content/ICCV2021/html/Shedligeri_SeLFVi_Self-Supervised_Light-Field_Video_Reconstruction_From_Stereo_Video_ICCV_2021_paper.html
    webpage: https://asprasan.github.io/pages/webpage-ICCV/index.html

    The network takes as input a sequence of stereo frames and then outputs
    a corresponding 4D light-field frame for each input stereo frame
    '''
    device = torch.device(
        f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    utils.setup_experiment(args)
    utils.init_logging(args)

    # indices of the input stereo frame in the output LF
    # the code currently supports only odd number of angular views
    left_view_idx = int(args.angular * (args.angular // 2))
    right_view_idx = int(left_view_idx + (args.angular - 1))
    lf_view_idx = [left_view_idx, right_view_idx]
    print(f'using stereo view indices as {lf_view_idx}')

    trainable_params = [] # a list for all trainable parameters
    models_list = [] # a list of all trainable models

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

    logging.info(
        f"Built {len(models_list)} models consisting of {sum(p.numel() for p in trainable_params):,} parameters")

    # ========== Initialize the intermediate low-rank representation model ==============
    # This model is adapted from: Wetzstein, Gordon, et al. "Tensor displays: compressive light field synthesis using multilayer displays with directional backlighting." (2012).
    if args.display == 'multilayer':
        tensor_display = models.multilayer(
            args.angular,# number of angular views in U/V
            args.layers, # number of layers in the low-rank model (default=3)
            args.inph, # height of the image
            args.inpw, # width of the image
            args=args).to(device)
    else:
        print('No valid display type chosen')
        print('exiting')
        exit(0)
    logging.info(
        f"Using the {args.display} display with {args.layers} layers and {args.rank} rank")

    criterion = torch.nn.MSELoss()

    # define model optimizer and the learning rate scheduler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=args.patience, min_lr=1e-6)

    # if resume training, then load pre-trained weights into each of the models
    if args.resume_training:
        state_dict = utils.load_checkpoint(
            args, models_list, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = state_dict['epoch']
        for g in optimizer.param_groups:
            g['lr'] = args.lr
    else:
        global_step = -1
        start_epoch = 0

    # initialize train, val and test data loaders
    # the output of the dataloader is just a batch of stereo frames
    # being a self-supervised algorithm, there's no ground truth LF data
    # args.data_path contains the absolute path to the h5 file containing the test frames
    # args.dataset calls different dataloaders that can help pre-process different datasets differently
    train_loader, valid_loader, test_loader = data.build_dataset(
        args.dataset, args.data_path, args, batch_size=args.batch_size, num_workers=4)

    # Track moving average of loss values
    # and initialize the tensorboard summary writer for logging
    train_meters = {name: utils.RunningAverageMeter(
        0.98) for name in (["train_loss"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_loss"])}
    test_meters = {name: utils.AverageMeter() for name in (["test_loss"])}
    writer = SummaryWriter(
        log_dir=args.experiment_dir) if not args.no_visual else None

    # initialize the consistency loss functions
    # geometric loss function
    loss_geo = utils.loss_geo_stereo_disp(args=args)
    # temporal consistency loss function
    loss_temp = utils.temporal_criterion(args=args)
    # smoothness consistency losses for depth, flow and image
    tv_smooth_loss = utils.sm_loss(args=args)
    # photometric consistency function is just a simple L1 loss between two frames
    # hence, no special implementation for it

    def run_batch(inputs, outputs, is_train=True):
        optimizer.zero_grad()
        # initialize a dictionary of inputs and outputs
        # that will help keep track of ouputs across timesteps
        inputs['video'] = inputs['video'].to(device)
        inputs['states_disp'] = [None, ] * 2 # initial states of LSTM network
        inputs['states_flow'] = [None, ] * 2
        inputs['lf_states'] = None
        inputs['flow_loss'] = False # changed to True after 1 time-step
        inputs['prev_idx'] = 0
        inputs['curr_step'] = 0
        outputs['loss'] = 0.
        outputs['step_loss'] = 0.
        outputs['pred_lf'] = []
        outputs['pred_disp'] = []
        outputs['pred_flow'] = []

        for t in range(1, inputs['video'].size(1)):
            # first do for the lf_model
            if t > 0:
                inputs['flow_loss'] = True
                inputs['prev_idx'] = int(t - 1)
            inputs['curr_step'] = t
            run_instance(inputs, outputs)
            outputs['loss'] += outputs['step_loss']
        if is_train:
            outputs['loss'].backward()
            optimizer.step()

    def run_instance(inputs, outputs):
        # [N,2,2,3,H,W]: targets
        # [batch, time, view, rgb, height, width]: description of the 'targets' dimensions
        targets = inputs['video'][:, inputs['prev_idx']:inputs['curr_step'] + 1, ...]
        curr_stereo_frame = targets[:, -1, ...]
        instance_loss = 0.

        # first predict disparity/depth
        N, _, _, _, h, w = targets.size()
        # outputs['pred_disp'] is a list of lenth two 
        # consisting of disparity map each for left and right frames
        outputs['pred_disp'] = disp_model(curr_stereo_frame, inputs)

        # compute the low-rank intermediate representation with stereo frames as input
        decomposition = lf_model(curr_stereo_frame, inputs)
        # decomposition is of size [N,layers,rank,3,h,w]
        outputs['pred_lf'].append(tensor_display(decomposition))

        # L_geo, L_stereo and L_disp in Eq. (12) of the paper 
        # are bundled into a single function below
        lf_consis = loss_geo.consistency(
            outputs['pred_lf'][-1], curr_stereo_frame, outputs['pred_disp'])

        # smoothness losses
        depth_sm = tv_smooth_loss.edge_aware_loss(
            outputs['pred_disp'], curr_stereo_frame)

        instance_loss += lf_consis + args.lambda_sm * (depth_sm)

        # temporal regularization between stereo light field views 
        # and optical flow
        if inputs['flow_loss']: # flag is true from the second time step onwards
            # first compute optical flow between successive stereo frames
            outputs['pred_flow'] = flow_model(targets, inputs)
            # temporal_loss = loss_temp.consistency(all[-2:][:,lf_view_idx,...],pred_flow,detach=True)
            # temporal loss between the ground truth frames as well
            # [N,2,2,3,H,W]: targets
            # [batch, time, rgb, height, width]
            gt_stereo = [targets[:, k, ...] for k in range(targets.size(1))]
            temporal_loss = loss_temp.consistency(
                gt_stereo, outputs['pred_flow'])
            instance_loss += args.lambda_temp * temporal_loss
            instance_loss += args.lambda_sm * \
                tv_smooth_loss.flow_sm(outputs['pred_flow'])
        outputs['step_loss'] = instance_loss

    for epoch in range(start_epoch, args.num_epochs):
        if args.resume_training:
            # if starting from an existing checkpoint
            if epoch % 50 == 0:
                factor = 1.1
                optimizer.param_groups[0]["lr"] /= factor
                print(f'learning rate reduced by factor of {factor}')

        train_bar = utils.ProgressBar(train_loader, epoch)

        for meter in train_meters.values():
            # reset the moving average loss trackers
            meter.reset()

        for model in models_list:
            # set each of the learnable models to train mode in pytorch
            model.train()

        for batch_id, inputs in enumerate(train_bar):
            # targets are a batch of RGB stereo frames
            # as it's a stereo video, it has a sequence of T frames
            # so, the input size will be [N,T,2,3,H,W]
            global_step += 1
            outputs = {}
            run_batch(inputs, outputs)
            
            # update the loss value at each iteration.
            # train meter does a moving average of these loss values
            train_meters["train_loss"].update(outputs['loss'].item())

            # log to the log file
            train_bar.log(
                dict(
                    **train_meters,
                    lr=optimizer.param_groups[0]["lr"]),
                verbose=True)

            # log values to the tensorboard
            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar(
                    "lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar(
                    "loss/train",
                    train_meters["train_loss"].avg,
                    global_step)
                sys.stdout.flush()

        if epoch % args.valid_interval == 0:
            # validation epoch...
            for model in models_list:
                model.eval()

            for meter in valid_meters.values():
                meter.reset()

            valid_bar = utils.ProgressBar(valid_loader)
            for sample_id, inputs in enumerate(valid_bar):
                with torch.no_grad():
                    outputs = {}
                    run_batch(inputs, outputs, is_train=False)

                    val_loss = outputs['loss'].item()
                    valid_meters["valid_loss"].update(val_loss)

                    if writer is not None and sample_id < 1:
                        # pred_lf is of size [N,25,C,h,w]
                        # for add_video [N,T,C,H,W]
                        pred_lf = outputs['pred_lf'][-1]
                        n_data_to_log = 2
                        writer.add_video(
                            f"valid_samples/pred/{sample_id}", pred_lf[:n_data_to_log], global_step)
                        writer.add_video(
                            f"valid_samples/gt/{sample_id}", inputs['video'][:n_data_to_log, -1, ...], global_step)

                        pred_depth1 = outputs['pred_disp'][0]
                        pred_depth2 = outputs['pred_disp'][1]
                        depth = torchvision.utils.make_grid(
                            pred_depth1, nrow=2, normalize=True)
                        writer.add_image(
                            f"valid_samples/depth1/{sample_id}", depth, global_step)
                        depth = torchvision.utils.make_grid(
                            pred_depth2, nrow=2, normalize=True)
                        writer.add_image(
                            f"valid_samples/depth2/{sample_id}", depth, global_step)
                        flow_np = outputs['pred_flow'][0].data.cpu().numpy()
                        flow_img = utils.flow2img(flow_np)
                        flow_img = np.transpose(flow_img / 255., [0, 3, 1, 2])
                        flow = torchvision.utils.make_grid(
                            torch.FloatTensor(flow_img), nrow=2, normalize=True)
                        writer.add_image(
                            f"valid_samples/flow/{sample_id}", flow, global_step)

            if writer is not None:
                writer.add_scalar(
                    "loss/valid",
                    valid_meters['valid_loss'].avg,
                    global_step)
                sys.stdout.flush()

            # test loader
            for meter in test_meters.values():
                meter.reset()

            test_bar = utils.ProgressBar(test_loader)
            for sample_id, inputs in enumerate(test_bar):
                with torch.no_grad():
                    outputs = {}
                    run_batch(inputs, outputs, is_train=False)
                    test_loss = outputs['loss'].item()
                    test_meters["test_loss"].update(test_loss)

            if writer is not None:
                writer.add_scalar(
                    "loss/test",
                    test_meters['test_loss'].avg,
                    global_step)
                sys.stdout.flush()

            logging.info(
                train_bar.print(
                    dict(
                        **train_meters,
                        **valid_meters,
                        **test_meters,
                        lr=optimizer.param_groups[0]["lr"])))
            utils.save_checkpoint(
                args,
                global_step,
                epoch,
                models_list,
                optimizer,
                score=valid_meters["valid_loss"].avg,
                mode="min")
            scheduler.step(valid_meters["valid_loss"].avg)

    logging.info(
        f"Done training! Best ACC {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument(
        "--data-path",
        default="data",
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
        help='spatial resolution of the light field')
    parser.add_argument(
        '--inpw',
        type=int,
        default=128,
        help='spatial resolution of the light field')
    parser.add_argument(
        '--angular',
        type=int,
        default=5,
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
    parser.add_argument(
        "--test",
        action="store_true",
        help="if true, then use the RGB test data")
    parser.add_argument(
        "--gt_loss",
        action="store_true",
        help="if true, then use the GT for supervision")
    parser.add_argument(
        "--psnr",
        action="store_true",
        help="if true,then compute PSNR from GT")
    # Add loss parameters
    parser.add_argument(
        "--lambda-sm",
        default=0.001,
        type=float,
        help="how much to weight the TV smoothness loss")
    parser.add_argument(
        "--lambda-temp",
        default=0.01,
        type=float,
        help="how much to weight the TV smoothness loss")
    parser.add_argument(
        "--lambda-ssim",
        default=0.0,
        type=float,
        help="how much to weight the SSIM loss")
    parser.add_argument(
        "--lambda-depth",
        default=1.0,
        type=float,
        help="how much to weight the SSIM loss")
    parser.add_argument(
        "--metric",
        default='l2',
        type=str,
        help="whether to use perceptual or l2 metric")

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument(
        "--num-epochs",
        default=200,
        type=int,
        help="force stop training at specified epoch")
    parser.add_argument(
        "--patience",
        default=10,
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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"set gpu device to {args.gpu}")
    import logging
    import sys
    import torch
    import torchvision
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter

    import data
    import models
    import utils

    main(args)
