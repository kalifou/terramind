#!/bin/bash

for sensor_i in s1 s2; do
    for terramind_backbone_size in base large; do
	for freeze_backbone_i in 1 0; do
            sbatch login_node__terrabyte.sh $sensor_i $terramind_backbone_size $freeze_backbone_i
	done
    done
done

