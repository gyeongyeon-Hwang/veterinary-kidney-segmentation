import json
import os
import random
import shutil
import numpy as np
# from batchgenerators.utilities.file_and_folder_operations import *

import multiprocessing


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def split(base, out):
    cases = os.listdir(base)
    print(len(cases))
    maybe_mkdir_p(out)
    maybe_mkdir_p(os.path.join(out, "imagesTr"))
    maybe_mkdir_p(os.path.join(out, "imagesTs"))
    maybe_mkdir_p(os.path.join(out, "labelsTr"))
    maybe_mkdir_p(os.path.join(out, "labelsTs"))
    case_id_list = []
    num_train_case = 64
    num_test_case = len(cases) - num_train_case
    random.seed(1)
    random.shuffle(cases)
    train_case = []
    test_case = []
    file_name = []
    for idx, c in enumerate(cases):
        # print(len(cases))
        idx += 1
        # print(c)
        case_id = int(c.split(".")[0])
        print(idx, case_id)
        # case_id_list.append(case_id)
        # print(len(case_id_list))
        # if len(case_id_list) < num_train_case:
        if idx < num_train_case:
            train_case.append(c)
            file_name.append(f'{idx:03d}_'+c)
            shutil.copy(os.path.join(base, c), os.path.join(out, "imagesTr", f'{idx:03d}_'+c))
            shutil.copy(os.path.join(base.replace('input', 'label'), c), os.path.join(out, "labelsTr", f'{idx:03d}_'+c))
        else:
            test_case.append(c)
            file_name.append(f'{idx:03d}_'+c)
            shutil.copy(os.path.join(base, c), os.path.join(out, "imagesTs", f'{idx:03d}_'+c))
            shutil.copy(os.path.join(base.replace('input', 'label'), c), os.path.join(out, "labelsTs", f'{idx:03d}_'+c))
    print(len(train_case), len(test_case))
    #
    json_dict = {}
    json_dict['name'] = "jbnu_CT"
    json_dict['description'] = "kidney, kidney calculi, and kidney Felvis & Fats segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "Animal(dogs) CT data for MICCAI"
    json_dict['licence'] = ""
    json_dict['release'] = "1.0 2022/12/30"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Kidney",
        "2": "Calculi",
        "3": "Pelvis & Fats"
    }
    json_dict['evaluationClass'] = [
        1,
        2,
        3
    ]

    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "Kidney",
    #     "2": "Pelvis & Fats"
    # }
    # json_dict['evaluationClass'] = [
    #     1,
    #     2
    # ]


    json_dict['numTraining'] = num_train_case
    json_dict['numTest'] = num_test_case
    json_dict['training'] = [{'image': f"{out}/imagesTr/{i}", "label": f"{out}/labelsTr/{i}"} for i in file_name[:num_train_case]]
    json_dict['test'] = [{'image': f"{out}/imagesTs/{i}", "label": f"{out}/labelsTs/{i}"} for i in file_name[num_train_case:]]
    # json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
    #                          cases[:210]]
    # json_dict['test'] = ["./imagesTr/%s.nii.gz" % i for i in cases[210:]]
    #
    with open(os.path.join(out, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)
    # save_json(json_dict, os.path.join(out, "dataset.json"))

if __name__ == '__main__':

    base = "./dataset/nii_data_raw/PRE/input"
    # base = "./dataset/nii_raw/PRE/input"
    # base = "/hdd1/MICCAI/raw_data/PRE/input"
    # out = "/hdd1/MICCAI/rearrange_all"
    out = "./dataset/nii_rearrange"
    # split(base, out)



    proc = multiprocessing.Process(target=split, args=(base, out))
    proc.start()
    proc.join()