import datetime
import time
# from nilearn.image import resample_img
import nibabel as nib
from tqdm.auto import tqdm
from pandas import DataFrame
import natsort
import numpy as np
import h5py
import argparse
import os
from monai.transforms import Compose, CropForegroundd

def read_hdf(hdf_root):
    normal_abnormal = natsort.natsorted(os.listdir(hdf_root))
    file_path = []
    for state in normal_abnormal:
        path = os.path.join(hdf_root, state)
        hdf_list = natsort.natsorted(os.listdir(path))
        for hdf_name in hdf_list:
            hdf_path = os.path.join(path, hdf_name)
            file_path.append(hdf_path)
    print(f'# files: {len(file_path)}')
    return file_path

def npy_to_nii(volume, origin_spacing, case_id, contrast, save_directory, type='input'):

    origin_dtype = volume.dtype
    print(f'nii shape: {volume.shape}')
    origin_affine = np.diag(list(origin_spacing) + [1])
    print(origin_affine)
    volume_nifti = nib.Nifti1Image(volume, origin_affine)

    if type == 'input':
        nii_path = os.path.join(save_directory, f'{contrast}/input/{case_id}.nii.gz')
        nib.nifti1.save(volume_nifti, nii_path)
    else:
        nii_path = os.path.join(save_directory, f'{contrast}/label/{case_id}.nii.gz')
        nib.nifti1.save(volume_nifti, nii_path)




def make_folder(save_directory, contrast):
    input_label = ['input', 'label']
    for i_l in input_label:
        os.makedirs(os.path.join(save_directory, f'{contrast}/{i_l}/'), exist_ok=True)

def read_txt(file, hdf=True):
    d = []
    f = open(file, 'r')
    for line in f:
        if hdf == True:
            d.append(str(line.strip()) + '.hdf')
        else:
            d.append(str(line.strip()))
    return d

def main(hdf_path, save_directory):
    start = time.time()
    name_list = []
    hdf_list = []
    pre_post_list = []
    normal_abnormal_list = []
    label_class_before = []
    label_class_after = []
    before_shape_data = []
    before_shape_label = []
    after_shape_data = []
    after_shape_label = []
    hdf_in_name = []
    spacing_x_list = []
    spacing_y_list = []
    spacing_z_list = []
    hu_list = []
    clip_hu_list = []
    vol_data_shape_list = []
    vol_label_shape_list = []







    no_changed_list = ['ABNORMAL_70', 'ABNORMAL_82', 'ABNORMAL_94', 'ABNORMAL_104', 'ABNORMAL_105', 'ABNORMAL_111', 'ABNORMAL_112', 'ABNORMAL_115',
                       'ABNORMAL_116', 'ABNORMAL_130', 'ABNORMAL_135', 'ABNORMAL_136', 'ABNORMAL_137', 'ABNORMAL_144', 'ABNORMAL_190', 'NORMAL_51',
                       'NORMAL_73', 'NORMAL_76', 'NORMAL_77', 'NORMAL_79', 'NORMAL_86']
    # no_changed_list = read_txt(axis_txt, hdf=True)
    tmp_list = ['2876', '5864', '5921', '5923', '5984', '6039', '6074', '202100843', '16649', '16064', '16154', '13012', '13165', '13678', '13330', '13012',
                '12907', '202000220', '202001284', '16616', '16623', '202002704', '6896', '6494', '202002823', '202001387', '202100593', '202100570',
                '202100630', '15613', '6494', '16360', '14114', '16334', '16315', '16314', '16239', '11641', '16289', '5967', '11645', '11686', '202100842',
                '9664', '202001130', '202001251', '202002464', '13438', '6350', '202100919', '202001395', '3190', '6127', '6237', '13367']
    # tmp_list = read_txt(tmp_txt, hdf=False)
    file_path = read_hdf(hdf_path)

    with tqdm(total=len(file_path) - 1, ascii=True, desc=f'extract', dynamic_ncols=True) as pbar:
        # idx = 1
        for hdf in file_path:
            file_name = hdf.split('/')[-1]
            normal_abnormal = file_name.split('_')[0]
            f = h5py.File(hdf, 'r')
            num_image = f['ExportData']['number_of_image'][()][0]
            for image_idx in range(num_image):
                idx_name = 'Image_' + str(image_idx+1)
                data = f['ExportData'][idx_name]['image'][()].astype(np.float32)
                label = f['ExportData'][idx_name]['label'][()]

                img_name = str(f['ExportData'][idx_name]['name'][()][0])
                img_name = img_name.replace("b'", '')
                img_name = img_name.replace("'", '')
                img_name = img_name.replace(" ", '_')
                img_name = img_name.replace('-_', '')
                img_name = img_name.replace('@', '')
                img_name = img_name[:-1]
                if 'DORSAL' in img_name:
                    img_name = img_name.replace('DORSAL', 'POST')
                    print('8 : ' + img_name)

                img_split0 = img_name.split('_')

                if img_split0[0].find('\\') != -1:
                    case_id = img_split0[0].split('\\')
                    case_id = case_id[0]


                else:
                    if img_split0[0].find('@') == 1:
                        case_id = img_split0[0].replace('@', '')
                    else:
                        case_id = img_split0[0]
                img_split = []
                for i in img_split0:
                    img_split.append(i.upper())
                print(img_split)
                print(f'case name : {case_id}')

                label_num = np.unique(label)

                if 'POST' in img_split:
                    contrast = 'POST'
                elif 'PRE' in img_split:
                    contrast = 'PRE'
                elif 2 in label_num:
                    contrast = 'PRE'
                elif 3 in label_num:
                    contrast = 'POST'
                else:
                    contrast = 'POST'
                print(f'contrast: {contrast}')
                if contrast == 'POST' or os.path.isdir(os.path.join(save_directory,
                                                                    f'segmentation/{contrast}/input/{case_id}.npy')):
                    continue

                if case_id not in tmp_list:
                    try:
                        depth = f['ExportData'][idx_name]['depth'][()][0]
                    except KeyError:
                        depth = 1

                    height = f['ExportData'][idx_name]['height'][()][0]
                    width = f['ExportData'][idx_name]['width'][()][0]
                    print(f'data shape : {data.shape}')
                    data = np.reshape(data, (depth, height, width, int(data.shape[0] / (depth * height * width))))
                    print(f'data reshape : {data.shape}')
                    before_data_shape = data.shape

                    label = np.reshape(label, (depth, height, width))
                    print(f'label shape : {label.shape}')
                    before_label_shape = label.shape
                    # resample data
                    spacing = f["ExportData"][idx_name]["spacing"][:]
                    resampled_datas = []

                    hu_value = list(map(str, list(np.unique(data))))
                    hu_value = hu_value[:5] + ["..."] + (list(hu_value[-5:]))
                    hu_value = " ".join(hu_value)
                    data = data.squeeze()
                    #                 축이 뒤바뀌지 않은 파일들
                    if (file_name in no_changed_list) or (contrast == 'PRE') or (normal_abnormal == 'NORMAL'):
                        data = np.transpose(data, (2, 1, 0))
                        label = np.transpose(label, (2, 1, 0))
                    else:
                        data = np.transpose(data, (1, 0, 2, 3))
                        label = np.transpose(label, (1, 0, 2))
                        print(data.shape, label.shape)

                    label, label_record = convert_label(label, contrast)
                    if 2 in np.unique(label):
                        make_folder(save_directory, contrast)

                        # data = np.flip(data, 0)
                        # data = np.flip(data, 1)
                        data = np.flip(data, 2)

                        # print(spacing)
                        print(f'data spacing : {spacing}')

                        # label = np.flip(label, 0)
                        # label = np.flip(label, 1)
                        label = np.flip(label, 2)

                        WW, WL = 350, 40

                        data = np.clip(data, WL - (WW / 2), WL + (WW / 2))

                        clip_hu_value = list(map(str, list(np.unique(data))))
                        clip_hu_value = clip_hu_value[:5] + ["..."] + (list(clip_hu_value[-5:]))
                        clip_hu_value = " ".join(clip_hu_value)



                        print(np.unique(data))

                        # resample label
                        npy_to_nii(data, origin_spacing=spacing,
                                   case_id=case_id,
                                   contrast=contrast,
                                   save_directory=save_directory,
                                   )

                        npy_to_nii(label, origin_spacing=spacing,
                                   case_id=case_id,
                                   contrast=contrast,
                                   save_directory=save_directory,
                                   type='label'
                                   )

                        hu_list.append(hu_value)
                        clip_hu_list.append(clip_hu_value)
                        name_list.append(case_id)
                        hdf_in_name.append(img_name)
                        hdf_list.append(file_name)
                        pre_post_list.append(contrast)
                        normal_abnormal_list.append(normal_abnormal)
                        label_class_before.append(label_record[0])
                        label_class_after.append(label_record[1])
                        spacing_x_list.append(float(spacing[0]))
                        spacing_y_list.append(float(spacing[1]))
                        spacing_z_list.append(float(spacing[2]))

                        after_shape_data.append(data.shape)
                        after_shape_label.append(label.shape)
                        vol_data_shape_list.append(data.shape)
                        vol_label_shape_list.append(label.shape)

                        if max(np.unique(label)) >= 8:
                            # assert print(f'ID number[{case_id}]: labeling error -> {np.unique(label)}')
                            print(f'ID number[{case_id}]: labeling error -> {np.unique(label)}')
                            with open('./wrong_label.txt', 'w', encoding='utf-8') as f:
                                f.write(f'ID number[{case_id}]: labeling error -> {np.unique(label)}' + '\n')
                            break







                        df_dict = {'hdf': hdf_list,
                                   'case': name_list,
                                   'hdf in name': hdf_in_name,
                                   'Pre/Post': pre_post_list,
                                   'Normal/Abnormal': normal_abnormal_list,
                                   'label num before': label_class_before,
                                   'label num after': label_class_after,
                                   'spacing_x': spacing_x_list,
                                   'spacing_y': spacing_y_list,
                                   'spacing_z': spacing_z_list,
                                   'HU Value': hu_list,
                                   'Clip HU Value': clip_hu_list,
                                   'after reshape data(depth, height,width)': after_shape_data,
                                   'after reshape label(depth, height,width)': after_shape_label,
                                   'preprocessed data shape': vol_data_shape_list,
                                   'preprocessed label shape': vol_label_shape_list,
                                   }

                        df = DataFrame.from_dict(df_dict)
                        # os.makedirs(f"{save_directory}/", exist_ok=True)
                        df.to_excel(f"{save_directory}/{contrast}/data_table.xlsx")
                        # label_class.append(label_list)
                        print('---' * 20 + '\n')

                else:
                    print(f'비정상이여서 제외 : {case_id}')

                print('======================' * 3, '\n')
                # idx+=1
                pbar.update(1)

            runtime = time.time() - start
            print(f'HDF to png ---> {datetime.timedelta(seconds=runtime)}')

def converting(label, classes):
    classes_len = list(range(len(classes)))  # 0 ~ 7
    masks = [np.where(label == i) for i in classes_len]  # id:0 -> masks[0] ...
    mask_class = []  # [0 1 2]
    for i in classes:
        if i not in mask_class:
            mask_class.append(i)
    for mask, class_num in zip(masks, classes):
        label[mask] = mask_class.index(class_num)
    return label

def convert_label(label, contrast):
    """original label
        0: bg    1:  cortex  2: kidney   3: medulla  4: calculi  5: infarction
        6: cyst  7: None     8: pelvis & fat"""
    # label_list = []
    before = np.unique(label)
    label = label.astype(np.int8)
    label = np.where(label == 7, 0, label)
    label = np.where(label == 8, 7, label)
    """ 8->7
            0: bg    1:  cortex  2: kidney   3: medulla  4: calculi  5: infarction
            6: cyst  7: pelvis & fat"""

    if contrast == 'PRE':
        '''0: bg    1: kidney   2: calculi  3: pelvis & fat'''
        classes = [0, 1, 1, 1, 2, 1, 1, 3]
        label = converting(label, classes)
    else:
        '''0: bg   1: cortex   2: medulla   3: infarction   4: cyst'''
        classes = [0, 1, 2, 2, 2, 3, 4, 2]
        label = converting(label, classes)
    after = np.unique(label)
    # convert_l = f'{before} -> {after}'
    convert_l = [before, after]
    return label, convert_l

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--hdf_path', type=str, default='/hdd1/animal_ct/hdf_file/', help='hdf file path')
    parser.add_argument('--save_path', type=str, default='/hdd1/animal_ct_nii/raw_data_ww350_wl40',
                        help='save png file path')

    args = parser.parse_args()

    start = time.localtime()
    total_time = time.time()
    print('===' * 20)

    main(args.hdf_path, args.save_path)
