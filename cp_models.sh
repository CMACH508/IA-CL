#/bin/bash
num_models=25
run_id='tsp20_100k'

mkdir saved/SE-GNN/${run_id}/pretrained_models

pretrained_id=1
for model_id in $(seq 1 $num_models)
do
    for iter_id in $(seq 1 1)
    do
    cp saved/SE-GNN/$run_id/model${model_id}/models/pure_model_best_from_iter${iter_id}.pth saved/SE-GNN/$run_id/pretrained_models/model${pretrained_id}_pure_model_best.pth && \
    pretrained_id=`expr $pretrained_id + 1`
    done
done
