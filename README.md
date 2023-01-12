# IA-CL: A Deep Bidirectional Competitive Learning Method for Traveling Salesman Problem

![image](../../blob/main/overview.png)

## Environment

For environment configuration, see "conda_list.txt" and "pip_list.txt". The important ones are the versions of the following libraries.

```
python==3.6.12
pytorch==1.2.0
torchvision==0.4.0
cudatoolkit==10.0.130
torch-scatter==1.3.1
torch-sparse==0.4.0
torch-cluster==1.4.4
torch-spline-conv==1.1.0
torch-geometric==1.3.2
``` 

## Train Data & Test Data
Download the data and unzip it in the current folder.
```
https://pan.baidu.com/s/1lOuzloemPxU4DGTre6qsJQ 
Password: 9wif 
```

## Train

Adjust "config_train.json"

```
gpu_id: select gpu according to the number
arch-graph_size: 20/50/100 for TSP20/TSP50/TSP100 respectively
data_loader-graph_num: 100000/40000/20000 for TSP20/TSP50/TSP100 respectively
```

Adjust "run_train.sh"

```
num_models: Set according to computing resources
run_id: set arbitrarily
```

Run "run_train.sh"

```
source ./run_train.sh
```

Copy models with "cp_models.sh". Note that the path in the file is set according to the actual situation.

```
source ./cp_models.sh
```

## Test

Adjust "config_test.json"

```
gpu_id: select gpu according to the number
mode: greedy/sampling/beamsearch
arch-graph_size: 20/50/100 for TSP20/TSP50/TSP100 respectively
```

Adjust "run_test.sh"

```
num_models: Set according to computing resources
run_id: run_id set during training
```

Run "run_test.sh"

```
source ./run_test.sh
```

Run "test_cl.py". Pay attention to the args in the file.

```
python test_cl.py
-i: run_id set during training
-n: Number of cities for the test instance 20/50/100
-nm: The number of models used for testing. 
```

## Pretrained models
The trained model is in the following link

```
https://pan.baidu.com/s/1nGq0xHnJMpNfCz_MJNErAQ 
Password: t5jm
```
