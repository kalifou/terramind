#!/bin/bash

# Benchmark for TiM capabilities on the base model v1

# Single TiM modality

for tim_sensor_i in sen1, dem, ndvi, lulc; do
    sbatch login_node__terrabyte.sh $tim_sensor_i
done


for tim_sensor_i in dem, ndvi, lulc; do
    sbatch login_node__terrabyte.sh sen1","$tim_sensor_i
done

for tim_sensor_i_inner in sen1, dem, ndvi, lulc; do
    for tim_sensor_outter in sen1, dem, ndvi, lulc; do
        if [ $tim_sensor_i_inner != $tim_sensor_outter ]; then 
            sbatch login_node__terrabyte.sh $tim_sensor_i_inner","$tim_sensor_outter;
        fi
    done
done