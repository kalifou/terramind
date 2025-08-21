#!/bin/bash

# Benchmark for TiM capabilities on the base model v1

# Single TiM modality
for funetuning_sensor_i in s1 s2; do
    for tim_sensor_i in sen1 dem ndvi lulc; do
        sbatch login_node__terrabyte.sh $tim_sensor_i $funetuning_sensor_i
    done
    
     if [ funetuning_sensor_i == s1 ]; then 
         tim_sentinel_modality=sen2
     else
         tim_sentinel_modality=sen1
     fi

    for tim_sensor_i_inner_1 in $tim_sentinel_modality dem ndvi lulc; do
        
        sbatch login_node__terrabyte.sh $tim_sensor_i_inner_1 $funetuning_sensor_i
    
        for tim_sensor_i_inner_2 in $tim_sentinel_modality dem ndvi lulc; do
            
            # TIM with a list of 2 modalities
            if [ $tim_sensor_i_inner_1 != $tim_sensor_i_inner_2 ] ; then 
                sbatch login_node__terrabyte.sh $tim_sensor_i_inner_1","$tim_sensor_i_inner_2 $funetuning_sensor_i
            fi
            
            # TIM with a list of 3 modalities
            for tim_sensor_i_outer in $tim_sentinel_modality dem ndvi lulc; do
            
                # Making sure of the permutations of 3 modalities is free from redudant sensors (each much be unique)
                if [ $tim_sensor_i_inner_1 != $tim_sensor_i_inner_2 ] &&  [ $tim_sensor_i_inner_1 != $tim_sensor_i_outer ] && [ $tim_sensor_i_inner_2 != $tim_sensor_i_outer ]; then 
                    sbatch login_node__terrabyte.sh $tim_sensor_i_inner_1","$tim_sensor_i_inner_2","$tim_sensor_i_outer $funetuning_sensor_i
                fi
                
            done
        done
    done


done


