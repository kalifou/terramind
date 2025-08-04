#!/bin/bash

for sensor_i in s2; do
    for terramind_backbone_size in base large; do
        sbatch login_node__terrabyte.sh $sensor_i $terramind_backbone_size
    done
done
