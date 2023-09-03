import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
import os
import yaml


def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):
    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()

    spacing = imImage.GetSpacing()
    # spacing = spacing[::-1]
    print(f'spacing: {spacing}')
    origin = imImage.GetOrigin()

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    print(f'voluem shape: {npimg.shape}', f'HU value: {np.unique(npimg)}')
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s' % (save_path)):
        os.makedirs('%s' % (save_path))

    imImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)

    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)
    print(f'resampling: {sitk.GetArrayFromImage(re_img_xy).shape}')

    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)

    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)
    print(f'target spacing resampling: {sitk.GetArrayFromImage(re_img_xyz).shape}')



    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 64, 64])
    print(f'crop img, lab:{sitk.GetArrayFromImage(cropped_img).shape, sitk.GetArrayFromImage(cropped_lab).shape}')
    sitk.WriteImage(cropped_img, '%s/%s.nii.gz' % (save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz' % (save_path, name))


if __name__ == '__main__':

    for t in ['train', 'test']:
        if t == 'train':
            src_path = './dataset/nii_rearrange/imagesTr/'
            tgt_path = './dataset/nii_conversion/train/'
        else:
            src_path = './dataset/nii_rearrange/imagesTs/'
            tgt_path = './dataset/nii_conversion/test/'

        name_list = os.listdir(src_path)
        name_list = [name.split('.')[0] for name in name_list]

        # os.chdir(src_path)
        num_images = len(name_list)

        if not os.path.exists(tgt_path + 'list'):
            os.makedirs('%slist' % (tgt_path))
        with open('%slist/dataset.yaml' % tgt_path, 'w', encoding='utf-8') as f:
            yaml.dump(name_list, f)

        for name in name_list:
            print(name)
            img_name = name + '.nii.gz'

            img = sitk.ReadImage(src_path + '%s' % img_name)
            lab = sitk.ReadImage(src_path.replace("images", "labels") + '%s' % img_name)


            ResampleImage(img, lab, tgt_path, name, (0.5, 0.5, 1.25))
            print(name, 'done')


