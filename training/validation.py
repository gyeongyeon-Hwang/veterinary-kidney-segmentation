import os
import torch
import torch.distributed as dist
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_dice
import numpy as np
from .utils import concat_all_gather
import logging
from utils import is_master
from tqdm import tqdm
import nibabel as nib

from utils import (
    configure_logger,
    save_configure,
    AverageMeter,
    ProgressMeter,
)

def validation(net, dataloader, epoch, save_path, args):
    output_path = save_path
    net.eval()

    dice_list = []
    ASD_list = []
    HD_list = []
    for i in range(args.classes-1): # background is not including in validation
        dice_list.append([])
        ASD_list.append([])
        HD_list.append([])

    inference = get_inference(args)

    logging.info("Evaluating")

    with torch.no_grad():
        iterator = tqdm(dataloader)
        for idx, (images, labels, spacing, case) in enumerate(iterator):
            # spacing here is used for distance metrics calculation
            case = str(case).replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace("[", "").replace("]","")

            inputs, labels = images.float().cuda(), labels.long().cuda()
            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)

            pred, inf_label = inference(net, inputs, labels, args)

            if (epoch + 1) % 50 == 0:
                print(f'Saving predicted results for epoch {epoch + 1}')
                output_dir = os.path.join(output_path, f'predicted_results_epoch_{epoch + 1}')
                print(output_dir)
                os.makedirs(output_dir, exist_ok=True)

                _, predicted_mask = torch.max(pred, dim=1)
                predicted_mask_np = predicted_mask.detach().cpu().numpy().astype(np.float32).squeeze(0)
                inf_label = inf_label.detach().cpu().numpy().astype(np.int8)
                inf_label = np.squeeze(inf_label)

                affine_m = np.eye(4)
                affine_m[:3, :3] = np.diag(spacing.numpy().ravel())
                affine_m[:3, 3] = [0., 0., 0.]
                print(f'case:{case}, {predicted_mask_np.shape, labels.shape}\n,'
                      f' {np.unique(predicted_mask_np), np.unique(inf_label)}\n,'
                      f'spacing: {spacing}')


                predicted_mask_np = np.transpose(predicted_mask_np, (2, 1, 0))  # (W, H, D) 순서로 변경
                inf_label = np.transpose(inf_label, (2, 1, 0))  # (W, H, D) 순서로 변경
                input_image_nifti = nib.Nifti1Image(predicted_mask_np, affine=affine_m)
                input_label_nifti = nib.Nifti1Image(inf_label, affine=affine_m)
                nib.save(input_image_nifti, os.path.join(output_dir, f'predicted_mask_{case}.nii.gz'))
                nib.save(input_label_nifti, os.path.join(output_dir, f'predicted_mask_{case}_gt.nii.gz'))

            _, label_pred = torch.max(pred, dim=1)

            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)


            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging (HD and ASD computation for large 3D image is slow)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)

            # exclude background
            dice = dice.cpu().numpy()[1:]

            labels = labels.cpu().numpy()

            for cls in range(0, args.classes-1):
                if cls+1 in np.unique(labels):
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    ASD_list[cls].append(tmp_ASD_list[cls])
                    HD_list[cls].append(tmp_HD_list[cls])
                    dice_list[cls].append(dice[cls])

    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append(np.array(dice_list[cls]).mean())
        out_ASD.append(np.array(ASD_list[cls]).mean())
        out_HD.append(np.array(HD_list[cls]).mean())

    return np.array(out_dice), np.array(out_ASD), np.array(out_HD)




def validation_ddp(net, dataloader, epoch, fold_idx, args):

    net.eval()
    output_path = f'./save_pred/{fold_idx}'
    dice_list = []
    ASD_list = []
    HD_list = []
    unique_labels_list = []

    inference = get_inference(args)

    logging.info(f"Evaluating")

    with torch.no_grad():
        iterator = tqdm(dataloader) if is_master(args) else dataloader
        for (images, labels, spacing, case) in iterator:
            # spacing here is used for distance metrics calculation

            inputs, labels = images.float().cuda(), labels.long().cuda()

            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)

            pred, inf_label = inference(net, inputs, labels, args)
            _, label_pred = torch.max(pred, dim=1)

            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)


            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            # comment this for fast debugging. (HD and ASD computation for large 3D images are slow)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            tmp_dice_list, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            labels = labels.cpu().numpy()
            unique_labels = np.unique(labels)
            unique_labels =  np.pad(unique_labels, (100-len(unique_labels), 0), 'constant', constant_values=0)
            # the length of padding is just a randomly picked number (most medical tasks don't have over 100 classes)
            # The padding here is because the all_gather in DDP requires the tensors in gpus have the same shape

            tmp_dice_list = tmp_dice_list.unsqueeze(0)
            unique_labels = np.expand_dims(unique_labels, axis=0)
            tmp_ASD_list = np.expand_dims(tmp_ASD_list, axis=0)
            tmp_HD_list = np.expand_dims(tmp_HD_list, axis=0)

            if args.distributed:
                # gather results from all gpus
                tmp_dice_list = concat_all_gather(tmp_dice_list)

                unique_labels = torch.from_numpy(unique_labels).cuda()
                unique_labels = concat_all_gather(unique_labels)
                unique_labels = unique_labels.cpu().numpy()

                tmp_ASD_list = torch.from_numpy(tmp_ASD_list).cuda()
                tmp_ASD_list = concat_all_gather(tmp_ASD_list)
                tmp_ASD_list = tmp_ASD_list.cpu().numpy()

                tmp_HD_list = torch.from_numpy(tmp_HD_list).cuda()
                tmp_HD_list = concat_all_gather(tmp_HD_list)
                tmp_HD_list = tmp_HD_list.cpu().numpy()


            tmp_dice_list = tmp_dice_list.cpu().numpy()[:, 1:] # exclude background
            for idx in range(len(tmp_dice_list)):  # get the result for each sample
                ASD_list.append(tmp_ASD_list[idx])
                HD_list.append(tmp_HD_list[idx])
                dice_list.append(tmp_dice_list[idx])
                unique_labels_list.append(unique_labels[idx])

    # Due to the DistributedSampler pad samples to make data evenly distributed to all gpus,
    # we need to remove the padded samples for correct evaluation.
    if args.distributed:
        padding_size = dist.get_world_size() - (len(dataloader.dataset) % dist.get_world_size())
        for _ in range(padding_size):
            ASD_list.pop()
            HD_list.pop()
            dice_list.pop()
            unique_labels_list.pop()


    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append([])
        out_ASD.append([])
        out_HD.append([])

    for idx in range(len(dice_list)):
        for cls in range(0, args.classes-1):
            if cls+1 in unique_labels_list[idx]:
                out_dice[cls].append(dice_list[idx][cls])
                out_ASD[cls].append(ASD_list[idx][cls])
                out_HD[cls].append(HD_list[idx][cls])

    out_dice_mean, out_ASD_mean, out_HD_mean = [], [], []
    for cls in range(0, args.classes-1):
        out_dice_mean.append(np.array(out_dice[cls]).mean())
        out_ASD_mean.append(np.array(out_ASD[cls]).mean())
        out_HD_mean.append(np.array(out_HD[cls]).mean())

    return np.array(out_dice_mean), np.array(out_ASD_mean), np.array(out_HD_mean)
