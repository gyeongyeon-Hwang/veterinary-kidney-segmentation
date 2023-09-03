#!/bin/bash


python train.py --model fhformer --dimension 3d --dataset animal --batch_size 2 --unique_name fhformer_0903_noMSF --gpu 1 > ./logout/fhformer_0903_noMSF.out &
