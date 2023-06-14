#!/bin/bash


while getopts t:w:d flag
do
    case "${flag}" in
        t) run_type=${OPTARG};;
        w) weights=${OPTARG};;
        d) debug='-m pdb'
    esac
done
    

if [[$run_type == 'train']]
then
    echo 'Training'
    python $debug main.py --env_name=Failure --tree_depth=2 --run_type=train --total_timesteps=300000
else 
    python $debug main.py --env_name=Failure --tree_depth=2 --run_type=evaluate --saved_weights=$weights
fi