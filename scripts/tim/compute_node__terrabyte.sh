#!/bin/bash

source activate base
#conda init bash
conda activate gruenblick_terramind

list_tim_sensor_i=$1
finetuning_sensor=$2
backbone_size=base

max_epoch_i=2
batch_size=32
dataset=/dss/dsshome1/04/di38jul/swattainer/temporary_rene/biomassters/
log_dir=../../results/benchmkark_tim/

echo "list of backbones: "$list_tim_sensor_i

python ../../benchmarking_terramind_on_biomassters.py \
       -tst\
       -sd 1\
       -fzb \
       -sr s2 \
       -ltim $list_tim_sensor_i\
       -bcsz  $backbone_size \
       -me $max_epoch_i \
       -bs $batch_size \
       -pthd $dataset \
       -pthl  $log_dir >> tim_$list_tim_sensor_i"__frozen__"sen2_$(date +"__%Y_%m_%d__%H_%M_%S")"_.out" 
