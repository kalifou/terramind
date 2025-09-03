#!/bin/bash

#SBATCH --nodes=1                # node count
#SBATCH --cluster=hpda2 
#SBATCH --partition=hpda2_compute_gpu
#hpda2_testgpu

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1 
#SBATCH --time=08:00:00
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j

#SBATCH --mem=99000
#SBATCH --cpus-per-task=48



sensor_i=$1
terramind_backbone_size=$2
freeze_backbone_i=$3

srun bash compute_node__terrabyte.sh $sensor_i $terramind_backbone_size $freeze_backbone_i
