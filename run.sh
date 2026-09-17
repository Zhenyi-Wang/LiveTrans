#!/bin/bash
cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate trans

# 杀掉旧进程（client 会管理 server 生命周期）；限定本用户，避免误匹配 docker 容器里的同名进程
pkill -u "$USER" -f "python main.py" 2>/dev/null || true

python main.py
