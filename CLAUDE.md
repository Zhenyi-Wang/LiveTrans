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
# 服务化启动：幂等地在 tmux 'livetrans' 会话中运行 run.sh
# （开机自动执行：.bashrc 末尾自启块按 boot_id 每开机周期拉起一次，2026-09-17 起替代 systemd 用户单元）
./start.sh

# 前台直接跑（调试用）：conda trans 环境，pkill 清残留后 python main.py
bash run.sh
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

### 翻译子系统（frontend/server/）

数据链路：whisper client 识别结果 POST `/backend/api/listen` → 前端 Nitro 做矫正+翻译 → WS 广播字幕。

**confirmed 正式翻译（串行队列，segments.ts）**：
- `enqueueConfirmed` 入队即赋唯一 id（前端按 id 匹配 confirmed→update 替换，**广播 confirmed 前 id 必须已赋值**，时序敏感）
- drain 循环单飞消费：每批最多 `NUXT_CONFIRMED_BATCH_MAX`（默认 2）条，积压循环补齐；失败重试 2 次→拆单兜底→回退原文，字幕不断流
- **共享对话流（conversationHistory）**：每批成功后把 (user 原文, 模型原始返回) append 进对话流（append 后永不变），请求前缀严格递增——每次仅 miss 增量（上批 output + 本批输入），最大化前缀缓存命中
- 对话流窗口锯齿式（2026-09-14 起）：超 `NUXT_HISTORY_MAX_ROUNDS`(默认100) 轮**一次裁到** `NUXT_HISTORY_MIN_ROUNDS`(默认20) 轮并打 `[history]` 日志，前缀断裂每 ~(MAX-MIN) 批一次（旧实现逐条 shift 每批断一次，实测命中率仅 82%，见 docs/2026-09-14 分析）

**current 预览翻译（listen.ts + Segment.previewInput）**：
- 文本变化才触发（needBroadcast 去重）；异步 fire-and-forget 不阻塞 POST 响应（dispatch 消费速度取决于响应时间）
- **与 confirmed 共享同一条对话流前缀**（互相保温缓存），最后一条 user 携带积压未翻原文（最多 3 条只发 original）作补充上文 + 尾部"仅返回英文翻译"开关（开关**说明**在 system——两种调用逐字节一致；开关**取值**在 user 侧——不能动 system 否则前缀分叉）
- 并发限制 1（与 confirmed 的 1 相加 = LLM 总并发 2）；结果缓存 50 条防转录抖动；新鲜度按句子 `start` 判断（同句演进可广播，跨句才拦）
- 频率闸（2026-09-14 起）：距上次 LLM 发起 ≥`NUXT_PREVIEW_MIN_INTERVAL_MS`(默认3000ms, start-to-start) 才发起下一次；等待期新文本照旧进 pending 槽（最新者胜不变），previewCache 命中不占闸——实测限流前调用中位间隔仅 1.7s，中间版本多数白翻
- **最新者胜**（single-flight+合并，无排队）：in-flight 期间新 current 只记入 pending 槽（覆盖旧值=忽略中间版本），上一个完成后翻 pending 里最新的——天然背压永不积压，高延迟渠道（如 dss 2.4s）下吞吐全部有效

**LLM 调用（ai.ts）**：
- 统一走 new-api 网关的 **Anthropic 端点 `/v1/messages`** + `thinking:{"type":"disabled"}`。当前模型 `dss/deepseek-v4-flash`（2026-09-09 试用：快 ~1.5s、缓存稳、无思考）；渠道对比与 gpt-5.6-luna 备选方案见 [docs/2026-09-09_翻译模型渠道对比与gpt思考缓存实测.md](docs/2026-09-09_翻译模型渠道对比与gpt思考缓存实测.md)。OpenAI 入口的思考控制参数会被 new-api 转换层丢弃，必须用原生 thinking 参数（注意：对 CPA 渠道 gpt 系无效，禁不掉思考）
- 渠道前缀：`go/`=ollama pro（并发上限 3，缓存统计恒 0）、`dss/`=快且缓存稳、无前缀模型名可能不存在（model_not_found）；`[usage]` 日志含 `think` 字段（gpt 系渠道）
- fetch 超时默认 15s、`NUXT_LLM_TIMEOUT_MS` 可调（防网关挂死卡住 drain）；aiQuery 外有全局并发信号量兜底

**后端 dispatch（whisper_live/client.py）**：
- 独立队列线程 + 失败 5s 退避重试：前端重启/不可用时 WS 不断流、恢复后自动续传，**无需重启后端**
- 过旧（>15s）的纯 current 直接丢弃加速重放；confirmed 永不丢

**主日值守监控（mini 侧 server/plugins/monitor.ts + home 侧转录检查端口）**：
- 默认主日（周日）07:40-09:25：开始时间+宽限(180s)后检查 转录可达/音频三态/字幕（最近30分钟回看），异常持续 60s 二次确认后经 tellme webhook 真实通知；结束时间+宽限(120s)后仍在直播则通知
- 架构：**mini 只拉 home 转录 `/status` 一个数据源**（run_client.py :9091 暴露流三态+WS+uptime，转录挂=拉取失败天然可检）；字幕读本进程 pipelineStatus（listen.ts 埋点，state 挂 globalThis 防 dev 模块双实例分裂）；值守是一次性通知，不做多源二次诊断
- 防误报：默认仅周日、宽限期只观察、二次确认、每检查点当天只通知一次；结束时转录不可达则不重复通知（开始检查已报过）
- env 见 [docs/2026-09-14_主日值守监控设计.md](docs/2026-09-14_主日值守监控设计.md)（`MONITOR_*` mini 侧 / `LIVETRANS_STATUS_PORT` 等 home 侧；`MONITOR_DRY_RUN=1` 测试不发送）；home 检查端口已生效（2026-09-14 重启），mini 侧 sync-mini.sh 部署即激活（`MONITOR_ENABLED=0` 关闭）

### 部署（mini）
- `sync-mini.sh`：build → rsync（.env/docker-compose/.output）→ `docker compose up -d --force-recreate`
- **必须 force-recreate**：.output 是挂载卷，内容更新不触发 compose 重建，不强制重启则容器跑旧代码
- 环境变量用 `NUXT_` 前缀（`NUXT_OPENAI_MODEL` 等）实现运行时覆盖；裸 `OPENAI_*` 会被 build 烘焙且运行时不生效
- 本地测试模式：改 `run_client.py` 的 `dispatch_api` 字面量为 `http://localhost:8081` 并重启后端，配合本地 `yarn dev`（tmux 会话 livetrans-fe）。**注意**：DISPATCH_API 环境变量方案存在未定位的失灵问题（进程 environ 值正确但 POST 从未发出，strace 零 connect；同值字面量正常），悬案待查，勿用 env 方式。dev 热重载频繁改动后可能崩（`#internal/nuxt/paths` 错误），删 `.nuxt` 重启即可，client 的 dispatch 自愈能扛住。**观测实时输出以 tmux pane 为准**，`tee` 的 dev.log 有管道缓冲滞后

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

## docs 知识索引

- [livetrans一直重启与Ctrl+C停不下排查](docs/2026-09-17_livetrans一直重启与Ctrl+C停不下排查.md) — 2026-09-17三因叠加(main.py监督循环无限拉起+旧信号处理只等不强杀+tmux服务器段错误全灭)、main.py修复(5s优雅/12s强杀/二连C-c立即/退出兜底清9090)、警示:tmux服务器住livetrans.service cgroup,stop单元=全tmux陪葬,停服务用tmux kill-session
- [主日值守监控设计](docs/2026-09-14_主日值守监控设计.md) — 2026-09-14 v2定稿: mini侧监控只拉home转录/status单源(转录挂=拉取失败天然可检)+本地字幕流水(globalThis防dev双实例)、宽限+二次确认防误报、真实推流+真实tellme实测矩阵、env配置表、home已生效/mini待部署
- [缓存命中率82%结构分析与日志时区修复](docs/2026-09-14_缓存命中率82%结构分析与日志时区修复.md) — 2026-09-13命中率82.26%为20轮饱和窗口的结构性稳态(每批append+shift断前缀,每代首条全量重发~1530tok)、本地日志与平台侧分毫对账、docker logs -t恒UTC需+8、[usage]已加ts字段
- [翻译模型渠道对比与gpt思考缓存实测](docs/2026-09-09_翻译模型渠道对比与gpt思考缓存实测.md) — 2026-09-09定档dss/deepseek-v4-flash试用、gpt-5.6-luna三坑(思考禁不掉/缓存不稳/延迟3~10s)、回切优化路径(prompt_cache_key+CPA源码机制)、网关渠道表
- [直播翻译挤压排查与LLM超时调优](docs/2026-09-07_直播翻译挤压排查与LLM超时调优.md) — 2026-09-06挤压根因(LLM上游劣化+60s超时占槽)、耗时实测(p99=8.5s)、超时降至15s可配置(NUXT_LLM_TIMEOUT_MS)、网关实际指向144.24.9.183待确认