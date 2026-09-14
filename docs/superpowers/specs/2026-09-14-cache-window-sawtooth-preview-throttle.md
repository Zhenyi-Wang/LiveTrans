# Spec: 对话流锯齿窗口 + current 预览限流

日期: 2026-09-14
状态: 待审核

## 背景

2026-09-13 礼拜实测（详见 [docs/2026-09-14_缓存命中率82%结构分析与日志时区修复.md](../../2026-09-14_缓存命中率82%结构分析与日志时区修复.md)）：

1. `segments.ts` 的历史窗口截断实现为逐条 `shift()`（每批 confirmed 成功后移除最旧一轮），窗口全天饱和在 20 轮 → **每批都从头上打断前缀**，DeepSeek 前缀缓存每批失效一次，每代首条请求全量重发 ~1500 tok。全天 6841 条请求命中率仅 82.26%，费用 ¥12.97/场。
2. preview 与 confirmed 并发双飞（LLM 并发=2），请求完成才写缓存，同代并发请求互相看不到前缀，额外 +~50% miss。
3. preview 无频率下限：礼拜进行中调用间隔中位数 1.7s（单飞循环翻完立刻发下一条），中间版本翻译多数被新版覆盖，纯浪费。

参数空间已用当日日志重放模拟器充分扫描（`/tmp/sim_cache2.py`，用户确认）：锯齿窗口 **20/100** + 预览限流 **3s**。

## 目标

### 1. 对话流锯齿窗口（frontend/server/utils/segments.ts）

- 窗口单位：轮（1 轮 = 1 条 user + 1 条 assistant 消息，即现状 `HISTORY_MAX_ROUNDS` 语义）。
- 追加逻辑不变：每批 confirmed 成功后 push 一轮（user 原文 + 模型原始返回，append 后不变）。
- 截断逻辑改为**锯齿式**：当轮数 **> HISTORY_MAX_ROUNDS(默认 100)** 时，**一次性**裁剪到保留最近 **HISTORY_MIN_ROUNDS(默认 20)** 轮（原地 `splice` 从头部移除），而非逐条 shift。
- 裁剪发生时输出一条独立日志（如 `[history] ...`），便于部署后观测断裂事件；不属于 `[usage]` 格式（不受"非目标"限制）。
- 参数在使用点做 `Number()` 收敛（运行时覆盖进的是字符串，按 `ai.ts` 的 `llmTimeoutMs` 先例，不依赖隐式转换）；MAX 被 env 配得比 MIN 小时钳到 MIN。
- 效果：前缀断裂频率从"每批一次"降为"每 (MAX-MIN) 批一次"（~每 80 批一次）。
- 参数可 env 覆盖：`NUXT_HISTORY_MIN_ROUNDS`、`NUXT_HISTORY_MAX_ROUNDS`（runtimeConfig 键 `historyMinRounds`/`historyMaxRounds`；`historyMaxRounds` 键已存在，默认值从 20 改 100）。

### 2. current 预览限流（frontend/server/routes/backend/api/listen.ts）

- 在 `previewCurrent` 单飞循环内，**实际发起 LLM 调用前**加频率闸：距上次 LLM 发起时刻不足 `previewMinIntervalMs`（默认 3000，env `NUXT_PREVIEW_MIN_INTERVAL_MS`）则 sleep 等满再发。
- 计时口径：**start-to-start**（距上次发起，非结束后计时）。
- 语义约束：
  - **最新者胜不变**：等待期间新 current 照旧进 pending 槽（`previewActive` 分支）；闸睡满后翻译的是**睡眠前已持有的 seg**，pending 里的更新文本由循环下一轮（同样过闸）接力翻译——不改变现有单飞循环的持有语义。
  - **previewCache 命中不占闸**：缓存命中（纯内存复用）不更新 `lastPreviewStart`，不受限流影响。
  - 单飞结构、失败回退原文、`broadcastIfFresh` 新鲜度判断均不变。
- 效果：preview LLM 调用从 4659 次/场降至 ~3050 次（-35%），屏上英文刷新节奏上限 1 次/3s。

## 非目标

- 不改 confirmed 批量大小、模型渠道、retry/fallback 逻辑。
- 不改 `[usage]` 日志格式。
- 不做并发调度精细化（同代让路等）。

## 预期效果（模拟口径，2026-09-13 数据重放）

- 命中率 82.26% → ~98.3%
- 费用 ¥12.97/场 → ~¥4.1（并发修正后 ~¥4.7）
- preview 调用 4659 → ~3048

## 验收标准

1. `yarn build` 构建通过。
2. 本地集成测试（小参数 env：如 MAX=15/MIN=5/间隔=1500ms）通过：
   - 连续 confirmed 批次驱动轮数增长，日志可见裁剪事件（`[history]` 日志 + messages dump 条数骤降）；裁剪后**第一条**请求 read ≈ system 段（最低点），其后逐批恢复增长至 system+MIN 轮水平；
   - 快速连发 current，preview `[usage]` 行发起间隔 ≥ 设定值，且条数符合 1/间隔 上限；
   - 裁剪后翻译链路正常出结果（broadcast 不中断）。
3. `sync-mini.sh` 部署后 mini 容器健康（Listening、TZ=CST、env 未覆盖时代码默认值生效）。

## 风险与边界

- 裁剪是破坏性操作：被裁掉的 80 轮历史不可恢复，之后请求的上下文只有最近 20 轮——已确认接受（模拟显示费用谷底与上下文质量的平衡点，用户拍板 20/100）。
- MAX-MIN=80 批 ≈ 全场 2123 批 → ~26 次断裂，每次断裂一代首条全量重发（~90 轮 × ~68 tok ≈ 6K tok），已计入模拟。
- 部署时机：非礼拜时段（当前空闲），容器重建无影响。
