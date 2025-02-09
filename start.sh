#!/bin/bash
# 初始化 conda
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate trans

# 杀掉所有名为 "python main.py", "python run_server.py", "python run_client.py" 的进程
pkill -f "python main.py"
pkill -f "python whisperlive/run_server.py"
pkill -f "python whisperlive/run_client.py"

# 运行 python main.py
python main.py
