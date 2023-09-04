# Parallel Spatial-Frequency Hybrid Network (PSFH-Net)
This repository contains code and data files to reproduce the segmentation results of a pre-contrast CT of a dog's kidney, kidney stone, and kidney pelvis.
![fig1](https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation/assets/76763306/ad5daa19-3406-4c3b-8653-242bfa3b7473)
## Requirement
    $ git clone https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation
    $ pip install -r requirement.txt
## Datasets request
![axial](https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation/assets/76763306/b2edafb8-84e2-4703-a053-8416eb973c2c) ![dorsal](https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation/assets/76763306/bac136f5-088f-4f54-90fd-fb59b6d3c597)

- Please request the data through the link below 
  (https://docs.google.com/forms/d/e/1FAIpQLSfx2Aj17ixdHmeMbbJofYhs3pkqj_f8AeCHVlYoDFbOzvPUdA/viewform)
## Experiment environment
- Ubuntu 22.04
- CUDA 11.7
- Python 3.8.17
- PyTorch 1.13.1
## Datasets processing
- Create a folder under {LOCAL_PATH} ./dataset
- Data Conversion
    $ python cd ./dataset_conversion
    $ python ./rearrange.py
    $ python ./animal3d.py
## Training

    $ python ./train.py --dataset {dataset_root} --batch_size {batch} --cp_path {checkpoint path} --log_path {log_path} --unique_name {experiment name} --gpu {NUM_DEVICE}

## Acknowledgements

## Reference
* This repository used the following code: (https://github.com/yhygao/CBIM-Medical-Image-Segmentation)
