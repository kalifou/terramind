#!/bin/bash

#SBATCH --nodes=1                # node count
#SBATCH --cluster=hpda2 
#SBATCH --partition=hpda2_compute_gpu

#hpda2_testgpu 

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1 
#SBATCH --time=01:00:00
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j

sensor_i=$1
terramind_backbone_size=$2

srun bash compute_node__terrabyte.sh $sensor_i $terramind_backbone_size
