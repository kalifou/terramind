#!/bin/bash

#SBATCH --nodes=1                # node count
#SBATCH --cluster=hpda2 
#SBATCH --partition=hpda2_testgpu

#hpda2_compute_gpu
#hpda2_testgpu 

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1 
#SBATCH --time=00:10:00
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j

list_tim_sensor_i=$1
finetuning_sensor=$2

srun bash compute_node__terrabyte.sh $list_tim_sensor_i $finetuning_sensor