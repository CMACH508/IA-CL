#!/bin/bash
num_models=20
run_id='tsp20_100k'

for model_id in $(seq 1 $num_models)
do
nohup python test.py -c config_test.json -i ${run_id} -m $model_id > ${model_id}.out 2>&1 &
done

