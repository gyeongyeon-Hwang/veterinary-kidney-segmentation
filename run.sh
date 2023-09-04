#!/bin/bash


nohup python train.py --model psfh_net --dimension 3d --dataset animal --batch_size 2 --unique_name psfh_net --gpu 0 > ./logout/psfh_net.out &
