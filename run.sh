#!/bin/bash


while getopts t:w:e:dc flag
do
    case "${flag}" in
        t) run_type=${OPTARG};;
        w) weights=${OPTARG};;
        d) debug='-m pdb';;
        c) cumulative='--is_cumulative_mode=true';;
        e) env_name=${OPTARG}
    esac
done
    

if [[ "$run_type" == "train" ]]
then
    echo 'Training'
    python $debug main.py --env_name=$env_name --tree_depth=2 --run_type=train --total_timesteps=300000 $cumulative
else 
    echo 'Testing'
    if [[ "$weights" == "" ]]
    then
        echo "No weight file specified, missing option -w"
    
    else
        echo "Using weights from $weights"
        python $debug main.py --env_name=$env_name --tree_depth=2 --run_type=evaluate --model_filename=$weights
    fi
fi