#!/bin/bash
#SBATCH --job-name="zh-gcdt-hfl-chinese-roberta"
#SBATCH --output="training_logs/log-MUL-train-zh-gcdt-hfl-chinese-roberta_%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

source activate py37
module list
exec 2>&1

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:13000
python MUL_main_Train.py --datapath=./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/  --use_org_Parseval True --epoch 50 --batch_size 1 --seed 111 
python MUL_main_Train.py --datapath=./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/  --use_org_Parseval True --epoch 50 --batch_size 1 --seed 222 
python MUL_main_Train.py --datapath=./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/  --use_org_Parseval True --epoch 50 --batch_size 1 --seed 333 
python MUL_main_Train.py --datapath=./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/  --use_org_Parseval True --epoch 50 --batch_size 1 --seed 444 
python MUL_main_Train.py --datapath=./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/  --use_org_Parseval True --epoch 50 --batch_size 1 --seed 555 
