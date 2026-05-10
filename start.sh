#!/bin/bash
cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate trans

# 杀掉旧进程（client 会管理 server 生命周期）
pkill -f "python main.py"

python main.py
