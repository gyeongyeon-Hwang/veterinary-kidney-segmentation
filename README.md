# Parallel Spatial-Frequency Hybrid Network (PSFH-Net)
This repository contains code and data files to reproduce the segmentation results of a pre-contrast CT of a dog's kidney, kidney stone, and kidney pelvis.
![fig1](https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation/assets/76763306/099b4ffb-746b-4b73-a7a5-635f4fb25b6c)
## Requirement
    $ git clone https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation
    $ pip install -r requirement.txt
## Datasets requestion
<img src="https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation/assets/76763306/96431289-658d-478a-9897-9c9f89327dfc" width="300" height="300"> <img src="https://github.com/gyeongyeon-Hwang/veterinary-kidney-segmentation/assets/76763306/45fab718-bf3e-43f7-87d5-9c329a15df62" width="300" height="300"></center> 

- Please request the data through the link below 
  (https://docs.google.com/forms/d/e/1FAIpQLSfx2Aj17ixdHmeMbbJofYhs3pkqj_f8AeCHVlYoDFbOzvPUdA/viewform)
## Experiment environment
- Ubuntu 22.04
- CUDA 11.7
- Python 3.8.17
- PyTorch 1.13.1
## Datasets processing
    $ python cd ./dataset_conversion
    $ python ./rearrange.py
    $ python ./animal3d.py

## Training

    $ python ./train.py --dataset {dataset_root} --batch_size {batch} --cp_path {checkpoint path} --log_path {log_path} --unique_name {experiment name} --gpu {NUM_DEVICE}

## Acknowledgements

## Reference
* This repository used the following code: (https://github.com/yhygao/CBIM-Medical-Image-Segmentation)
