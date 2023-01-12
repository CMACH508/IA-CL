#!/bin/bash
num_models=5
run_id='tsp20_100k'

mkdir $run_id

for model_id in $(seq 1 $num_models)
do
nohup python train.py -c config_train.json -i $run_id -m $model_id > ${run_id}/${run_id}_model${model_id}.log 2>&1 &
done
