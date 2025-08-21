#!/bin/bash

source activate base
#conda init bash
conda activate gruenblick_terramind

sensor_i=$1
backbone_size=$2
freeze_backbone_i=$3

max_epoch_i=2
batch_size=32
dataset=/dss/dsshome1/04/di38jul/swattainer/temporary_rene/biomassters/
log_dir=../../results/benchmkark/

if [ $freeze_backbone_i -gt 0 ]
then
    python ../../benchmarking_terramind_on_biomassters.py \
	   -tst \
	   -fzb \
	   -sr $sensor_i \
	   -bcsz  $backbone_size \
	   -me $max_epoch_i \
	   -bs $batch_size \
	   -pthd $dataset \
	   -pthl  $log_dir > $backbone_size"__frozen__"$sensor_i$(date +"__%Y_%m_%d__%H_%M_%S")"_.out"
else
    python ../../benchmarking_terramind_on_biomassters.py \
 	   -tst \
	   -sr $sensor_i \
	   -bcsz  $backbone_size \
	   -me $max_epoch_i \
	   -bs $batch_size \
	   -pthd $dataset \
	   -pthl  $log_dir > $backbone_size"__not_frozen__"$sensor_i$(date +"__%Y_%m_%d__%H_%M_%S")"_.out" 
fi
