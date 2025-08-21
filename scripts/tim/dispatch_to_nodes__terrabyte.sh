#!/bin/bash

# Benchmark for TiM capabilities on the base model v1

# Single TiM modality
for funetuning_sensor_i in s1 s2; do
    for tim_sensor_i in sen1 dem ndvi lulc; do
        sbatch login_node__terrabyte.sh $tim_sensor_i $funetuning_sensor_i
    done
done


funetuning_sensor_i=s2
for tim_sensor_i_inner in sen1 dem ndvi lulc; do
    for tim_sensor_outter in sen1 dem ndvi lulc; do
        if [ $tim_sensor_i_inner != $tim_sensor_outter ]; then 
            sbatch login_node__terrabyte.sh $tim_sensor_i_inner","$tim_sensor_outter $funetuning_sensor_i
        fi
    done
done

funetuning_sensor_i=s1
for tim_sensor_i_inner in sen2 dem ndvi lulc; do
    for tim_sensor_outter in sen2 dem ndvi lulc; do
        if [ $tim_sensor_i_inner != $tim_sensor_outter ]; then 
            sbatch login_node__terrabyte.sh $tim_sensor_i_inner","$tim_sensor_outter $funetuning_sensor_i
        fi
    done
done
