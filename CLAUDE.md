# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个实时语音转文字翻译系统，主要用于海宁夏沙基督教会的礼拜直播。项目包含前端Vue应用和后端WhisperLive语音识别服务。

## 架构结构

### 主要组件
- **frontend/** - Nuxt 3前端应用，提供实时翻译显示界面
- **whisperlive/** - WhisperLive语音识别服务，基于OpenAI Whisper模型
- **main.py** - 主服务进程管理脚本，监控GPU状态并自动重启服务

### 前端架构 (frontend/)
- 使用Nuxt 3框架构建
- 基于Vue 3 Composition API
- 使用Pinia进行状态管理
- 支持WebSocket实时通信
- 集成OpenCC进行简繁转换
- 支持中英文双语显示和同步滚动

### 后端架构 (whisperlive/)
- 基于WhisperLive实现实时语音转文字
- 支持Faster Whisper和TensorRT后端
- 提供WebSocket API接口
- 支持多种音频输入源（麦克风、音频文件、RTSP流等）

## 常用命令

### 前端开发
```bash
cd frontend
yarn install          # 安装依赖
yarn dev              # 启动开发服务器 (localhost:3000)
yarn build            # 构建生产版本
yarn preview          # 预览生产构建
```

### 后端服务
```bash
# 启动WhisperLive服务器
cd whisperlive
python run_server.py --port 9090 --backend faster_whisper

# 使用TensorRT后端（需要先构建TensorRT引擎）
python run_server.py --port 9090 --backend tensorrt --trt_model_path "/path/to/tensorrt/model"
```

### 完整服务启动
```bash
# 使用启动脚本（推荐）
./start.sh

# 或直接运行主进程
python main.py
```

### 环境依赖
```bash
# 激活conda环境
conda activate trans

# 安装WhisperLive依赖
cd whisperlive
pip install -r requirements/server.txt
```

## 开发注意事项

### 环境变量配置
前端需要配置以下环境变量（在frontend/nuxt.config.ts中）：
- `OPENAI_API_KEY` - OpenAI API密钥
- `OPENAI_BASE_URL` - OpenAI API基础URL
- `OPENAI_MODEL` - 使用的OpenAI模型

### WebSocket通信
- 前端通过WebSocket与后端WhisperLive服务通信
- WebSocket端口默认为9090
- 支持实时语音转文字结果传输

### GPU监控
- main.py包含GPU状态监控功能
- 当GPU进入P8状态时会自动重启服务
- 监控间隔为5秒，P8连续检测阈值默认为3次

### 前端组件结构
- `components/content/` - 内容显示相关组件
- `components/layout/` - 布局相关组件
- `components/common/` - 通用组件
- `composables/` - Vue组合式函数
- `pages/index.vue` - 主页面

### 语音识别配置
- 默认使用Whisper small模型
- 支持多语言识别
- 可配置是否启用语音活动检测(VAD)
- 支持翻译功能（翻译为英文）