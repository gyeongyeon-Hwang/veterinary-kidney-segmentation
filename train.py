import logging
import os
import random

import torch
import torch.nn as nn
import numpy as np
from model.utils import get_model
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from training.utils import update_ema_variables
from training.losses import DiceLoss
from training.validation import validation
from training.utils import (
    log_evaluation_result,
    get_optimizer,
    filter_validation_results
)
import yaml
import argparse
import time
import sys
import warnings

from model.psfh_net import PSFH_Net
from utils import (
    configure_logger,
    save_configure,
    AverageMeter,
    ProgressMeter,
    resume_load_optimizer_checkpoint,
    resume_load_model_checkpoint,
)
from training.dataset_animal import AnimalDataset
from thop import profile, clever_format

from torch.optim import lr_scheduler



warnings.filterwarnings("ignore", category=UserWarning)

os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

def train_net(net, args, ema_net=None, fold_idx=0):
    ################################################################################
    # Dataset Creation
    trainset = AnimalDataset(args, mode='train', k_fold=args.k_fold, k=fold_idx, seed=args.split_seed)

    trainLoader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(args.aug_device != 'gpu'),
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0)
    )

    testset = AnimalDataset(args, mode='test', k_fold=args.k_fold, k=fold_idx, seed=args.split_seed)

    testLoader = data.DataLoader(testset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    logging.info(f"Created Dataset and DataLoader")

    ################################################################################
    # Initialize tensorboard, optimizer and etc
    log_p = f"{args.log_path}{args.unique_name}/fold_{fold_idx}"
    writer = SummaryWriter(log_p)

    optimizer = get_optimizer(args, net)
    exp_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    if args.resume:
        resume_load_optimizer_checkpoint(optimizer, args)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda().float())
    criterion_dl = DiceLoss()

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ################################################################################
    # Start training
    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    for epoch in range(args.start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs}")


        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current lr: {current_lr:.4e}")
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, exp_scheduler, args)

        ########################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net

        # save the latest checkpoint, including net, ema_net, and optimizer
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
            'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_latest.pth")

        if (epoch + 1) % args.val_freq == 0:

            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, epoch, log_p, args)
            dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(dice_list_test, ASD_list_test, HD_list_test,
                                                                                    args)  # filter results for some dataset, e.g. amos_mr
            log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)

            if dice_list_test.mean() >= best_Dice.mean():
                best_Dice = dice_list_test
                best_HD = HD_list_test
                best_ASD = ASD_list_test

                # Save the checkpoint with best performance
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict() if not args.torch_compile else net._orig_mod.state_dict(),
                    'ema_model_state_dict': ema_net.state_dict() if args.ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")

            logging.info("Evaluation Done")

            logging.info(' '.join([f"Dice class{i}: {dice_list_test[i].mean():.4f}" for i in range(len(dice_list_test))]))
            logging.info(' '.join([f"HD class{i}: {HD_list_test[i].mean():.4f}" for i in range(len(HD_list_test))]))
            logging.info(' '.join([f"ASD class{i}: {ASD_list_test[i].mean():.4f}" for i in range(len(ASD_list_test))]))

            logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")

        # writer.add_scalar('LR', exp_scheduler, epoch + 1)
        writer.add_scalar('LR', current_lr, epoch + 1)

    return best_Dice, best_HD, best_ASD


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, exp_scheduler, args):
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    progress = ProgressMeter(
        len(trainLoader) if args.dimension == '2d' else args.iter_per_epoch,
        [batch_time, epoch_loss],
        prefix="Epoch: [{}]".format(epoch + 1),
    )

    net.train()

    tic = time.time()
    iter_num_per_epoch = 0
    for i, inputs in enumerate(trainLoader):
        img, label = inputs[0], inputs[1].long()
        if args.aug_device != \
                'gpu':
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

        step = i + epoch * len(trainLoader)  # global steps

        optimizer.zero_grad()

        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img)

                loss = 0

                if isinstance(result, tuple) or isinstance(result, list):
                    # if use deep supervision, add all loss together
                    for j in range(len(result)):
                        loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
                else:
                    loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
        else:
            result = net(img)
            loss = 0
            if isinstance(result, tuple) or isinstance(result, list):
                # If use deep supervision, add all loss together
                for j in range(len(result)):
                    loss += args.aux_weight[j] * (
                                criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))

            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)

            loss.backward()
            optimizer.step()

        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        epoch_loss.update(loss.item(), img.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

        writer.add_scalar('Train/Loss', epoch_loss.avg, epoch + 1)
    if args.amp:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            scaler.step(exp_scheduler)
    else:
        exp_scheduler.step()


def get_parser():
    parser = argparse.ArgumentParser(description='CBIM Medical Image Segmentation')
    parser.add_argument('--dataset', type=str, default='animal', help='dataset name')
    parser.add_argument('--model', type=str, default='fhformer', help='model name')
    parser.add_argument('--dimension', type=str, default='3d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile, only supported by pytorch2.0')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', help='if resume training from checkpoint')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')

    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    config_path = 'config/%s_3d.yaml' % (args.model)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s" % config_path)

    print('Loading configurations from %s' % config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def init_network(args):
    net = PSFH_Net(img_size=args.training_size, base_num_features=args.base_chan, num_classes=args.classes,
                     image_channels=1, num_only_conv_stage=args.num_only_conv_stage,
                     feat_map_mul_on_downscale=2, pool_op_kernel_sizes=args.pool_op_kernel_sizes,
                     conv_kernel_sizes=args.conv_kernel_sizes, deep_supervision=True,
                     max_num_features=args.max_num_features, depths=args.depths,
                     num_heads=args.num_heads,
                     window_size=args.window_sizes, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., dropout_p=0.1, drop_path_rate=0.2,
                     norm_layer=nn.LayerNorm, use_checkpoint=False, positional_encoding=args.positional_encoding,
                     stack=args.stack)

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None

    if args.resume:
        resume_load_model_checkpoint(net, ema_net, args)

    if args.torch_compile:
        net = torch.compile(net)
    return net, ema_net


if __name__ == '__main__':

    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')

    args.log_path = args.log_path + '%s/' % args.dataset

    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    Dice_list, HD_list, ASD_list = [], [], []

    for fold_idx in range(args.k_fold):

        args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
        os.makedirs(args.cp_dir, exist_ok=True)
        configure_logger(0, args.cp_dir + f"/fold_{fold_idx}.txt")
        save_configure(args)


        net, ema_net = init_network(args)

        net.cuda()

        '''count params'''
        rand_input = torch.randn(1, 1, args.training_size[0], args.training_size[1], args.training_size[2]).cuda()
        macs, params = profile(net, inputs=(rand_input,))
        print(net)
        # print(f"macs: {macs}  params:{params}")
        Flops = (macs / 2)
        macs, Flops, params = clever_format([macs, Flops, params], "%.3f")

        print(f"Flops: {Flops}  MAC:{macs}  params:{params}")
        logging.info(
            f"\n"f"GPU: {args.gpu},\n"
            + f"Unique name: {args.unique_name},\n"
            + f"Dataset: {args.dataset},\n"
            + f"Model: {args.model},\n"
            + f"Dimension: {args.dimension},\n"
            + f"Flops: {Flops}  MAC:{macs}  params:{params}"
        )
        if args.ema:
            ema_net.cuda()
        logging.info(f"Created Model")
        best_Dice, best_HD, best_ASD = train_net(net, args, ema_net, fold_idx=fold_idx)

        logging.info(f"Training and evaluation on Fold {fold_idx} is done")

        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)

    ############################################################################################3
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt", 'w') as f:
        np.set_printoptions(precision=4, suppress=True)
        f.write('Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\n")

        f.write("HD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {HD_list[i]}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\n")

        f.write("ASD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ASD_list[i]}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")

    print(f'All {args.k_fold} folds done.')

    sys.exit(0)
