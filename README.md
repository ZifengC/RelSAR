# UniSAR: Modeling User Transition Behaviors between Search and Recommendation
This is the official implementation of the paper "UniSAR: Modeling User Transition Behaviors between Search and Recommendation" based on PyTorch.


## Overview

The main implementation of UniSAR can be found in the file `models/UniSAR.py`. 


## Experimental Setting
All the hyper-parameter settings of UniSAR on both datasets can be found in files `config/UniSAR_KuaiSAR.yaml` and `config/UniSAR_Amazon.yaml`.
The settings of two datasets can be found in file `utils/const.py`.


## Quick Start

### 1. Download data
Download and unzip the processed data [Amazon](https://drive.google.com/file/d/1_YHVR7MfS9iJtcdmY75riNCZFY39fFLh/view?usp=drive_link) and [KuaiSAR](https://drive.google.com/file/d/1AgCl3Jd7UxJjGCOvUfx1Yf3SODUpyXiT/view?usp=drive_link). Place data files in the folder `data`.

### 2. Satisfy the requirements
The requirements can be found in file `requirements.txt`.

### 3. Train and evaluate our model:
Run codes in command line:
```bash
python3 main.py --model UniSAR --data KuaiSAR
```

### 3.1 Build a static top-k item graph
If you want a reusable static item graph for path-aware attention, run:
```bash
python3 utils/build_item_graph.py --data KuaiSAR --topk 16 --output data/item_graph_topk.pt
```
The builder prints progress bars for user traversal and top-k pruning. Then pass the graph path at training time:
```bash
python3 main.py --model UniSAR --data KuaiSAR --item_graph_path data/item_graph_topk.pt
```

### 4. Check training and evaluation process:
After training, check log files, for example, `output/KuaiSAR/logs/time.log`.


## Environments

We conducted the experiments based on the following environments:
* CUDA Version: 11.4
* OS: CentOS Linux release 7.4.1708 (Core)
* GPU: The NVIDIA® 3090 GPU
* CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
