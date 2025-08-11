#!/bin/bash

source activate base
#conda init bash
conda activate gruenblick_terramind

list_tim_sensor_i=$1
backbone_size=base

max_epoch_i=3
batch_size=32
dataset=/dss/dsshome1/04/di38jul/swattainer/temporary_rene/biomassters/
log_dir=../../results/benchmkark_tim/

python ../../benchmarking_terramind_on_biomassters.py \
-tst \
-fzb \      
-sr s2 \
-ltim $list_tim_sensor_i\
-bcsz  $backbone_size \
-me $max_epoch_i \
-bs $batch_size \
-pthd $dataset \
-pthl  $log_dir >> tim_$list_tim_sensor_i"__frozen__"$sensor_i$(date +"__%Y_%m_%d__%H_%M_%S")"_.out" 