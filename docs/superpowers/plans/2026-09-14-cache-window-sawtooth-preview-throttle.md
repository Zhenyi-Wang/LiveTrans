# Plan: 对话流锯齿窗口 + current 预览限流

日期: 2026-09-14
Spec: [docs/superpowers/specs/2026-09-14-cache-window-sawtooth-preview-throttle.md](../specs/2026-09-14-cache-window-sawtooth-preview-throttle.md)

## Task 1: runtimeConfig 新增参数（frontend/nuxt.config.ts）

`runtimeConfig` 块内,`historyMaxRounds` 行改为:

```ts
    // 共享对话流窗口(锯齿式): 超过 MAX 轮一次裁到 MIN 轮,拉长前缀断裂间隔以保 DeepSeek 缓存命中率
    historyMinRounds: Number(process.env.HISTORY_MIN_ROUNDS) || 20,
    historyMaxRounds: Number(process.env.HISTORY_MAX_ROUNDS) || 100,
    // current 预览 LLM 调用最小间隔(ms,start-to-start),限流中间版本翻译; NUXT_PREVIEW_MIN_INTERVAL_MS 可运行时覆盖
    previewMinIntervalMs: Number(process.env.PREVIEW_MIN_INTERVAL_MS) || 3000,
```

即: `historyMaxRounds` 默认 20→100; 新增 `historyMinRounds`、`previewMinIntervalMs`。env 裸名在 build 时生效,`NUXT_` 前缀运行时覆盖(项目既有约定)。

## Task 2: 锯齿窗口（frontend/server/utils/segments.ts）

2a. 常量区(第 97-98 行附近)改为:

```ts
const CONFIRMED_BATCH_MAX = runtimeConfig.confirmedBatchMax || 2;
// 使用点 Number() 收敛: NUXT_ 运行时覆盖进的是字符串(ai.ts llmTimeoutMs 先例); MAX 配小于 MIN 时钳到 MIN
const HISTORY_MIN_ROUNDS = Math.max(1, Math.trunc(Number(runtimeConfig.historyMinRounds)) || 20);
const HISTORY_MAX_ROUNDS = Math.max(HISTORY_MIN_ROUNDS, Math.trunc(Number(runtimeConfig.historyMaxRounds)) || 100);
```

2b. 截断逻辑(drainQueue 内,原 132-134 行 `while (...) shift()`)改为:

```ts
        // 锯齿式截断: 超过 MAX 轮一次裁到 MIN 轮(保留最近),而非逐条 shift——
        // 逐条截断每批都从头上打断前缀,DeepSeek 缓存全场失效(2026-09-13 实测命中率仅82%);
        // 拉长断裂间隔到 (MAX-MIN) 批一次,命中率 82%→~98%(见 docs/2026-09-14 分析)
        if (conversationHistory.length > HISTORY_MAX_ROUNDS * 2) {
          const cut = conversationHistory.length - HISTORY_MIN_ROUNDS * 2;
          conversationHistory.splice(0, cut);
          console.log(`[history] 锯齿裁剪: 移除${cut / 2}轮, 保留${HISTORY_MIN_ROUNDS}轮`);
        }
```

其余逻辑(push 追加、失败拆单兜底、广播)一律不动。

## Task 3: 预览限流（frontend/server/routes/backend/api/listen.ts）

3a. 模块顶部(单飞控制变量区)加:

```ts
// 预览频率闸: 距上次 LLM 发起不足该间隔则等满再发(start-to-start); 等待期新文本照旧进pending槽,
// 发起时翻闸前已持有的seg,pending更新文本由循环下一轮接力; previewCache命中不占闸(纯内存复用)
const PREVIEW_MIN_INTERVAL_MS = Math.max(0, Math.trunc(Number(useRuntimeConfig().previewMinIntervalMs)) || 3000);
let lastPreviewStart = 0
```

3b. `previewCurrent` 循环内,LLM 分支(`await seg.previewInput()` 前)加闸:

```ts
      } else {
        const wait = lastPreviewStart + PREVIEW_MIN_INTERVAL_MS - Date.now()
        if (wait > 0) await new Promise(r => setTimeout(r, wait))
        lastPreviewStart = Date.now()
        await seg.previewInput()
```

其余(cache 查询、broadcastIfFresh、pending 接力、finally 复位)一律不动。

## Task 4: 构建

```bash
cd /home/zhenyi/ownprojects/livetrans/frontend && yarn build
```

构建零报错即过。

## Task 5: 本地集成测试（小参数验证行为，再以默认参数部署）

测试用小参数跑真实链路（约 21 次 LLM 调用，~¥0.02，走生产网关 llm.346751.xyz）:

```bash
cd /home/zhenyi/ownprojects/livetrans/frontend
set -a; source .env; set +a
export NUXT_OPENAI_API_KEY="$OPENAI_API_KEY" NUXT_OPENAI_BASE_URL="$OPENAI_BASE_URL" NUXT_OPENAI_MODEL="$OPENAI_MODEL"
export PORT=8082 NITRO_PORT=8082
export NUXT_HISTORY_MAX_ROUNDS=15 NUXT_HISTORY_MIN_ROUNDS=5 NUXT_PREVIEW_MIN_INTERVAL_MS=1500
node .output/server/index.mjs > /tmp/lt.log 2>&1 &
```

驱动与断言(python 脚本, 测完 kill):

1. **锯齿窗口**: 串行 POST 18 批 confirmed（`{"confirmed":[{"start":i.0,"end":i+1.0,"text":"这是第i句测试文本，阿门。"}]}`)。POST 立即返回而 drain 异步(~1.5s/批), 故**轮询 /tmp/lt.log 中 confirmed `[usage]` 计数递增后再发下一批**(等效每批等 drain 完成)。断言:
   - confirmed `[usage]` 恰 18 条、0 err;
   - 第 16 批后出现 `[history] 锯齿裁剪` 日志恰 1 次;
   - messages dump 结构: 裁剪前 31 条消息(15轮×2+1), 裁剪后首批 11 条(5轮×2+1), 之后逐批 +2;
   - read 走时: 裁剪后首条 read ≈ 最低点(≈system+5轮), 其后逐批回升。
2. **预览限流**: 0.3s 间隔连发 6 个不同 text 的 current, 等 8s。断言:
   - preview `[usage]` ≤ 4 条(上限 = 1次/1.5s), 以 `ts − ms`(发起时刻)计算相邻发起间隔 ≥1.5s(ts 是完成时刻, 完成间隔受时长波动影响不可用);
   - current 中文广播不受影响(log 中 `current:` 行正常)。
3. 测后清理: kill 本地 node 进程。

## Task 6: 文档更新

- `CLAUDE.md` 翻译子系统两处:
  - "共享对话流"条目: 补锯齿语义(超 MAX=100 一次裁到 MIN=20, 断裂每 80 批一次, `NUXT_HISTORY_MIN_ROUNDS`/`NUXT_HISTORY_MAX_ROUNDS`);
  - "current 预览翻译"条目: 补 3s 频率闸(start-to-start, `NUXT_PREVIEW_MIN_INTERVAL_MS`, cache 命中不占闸)。
- `docs/2026-09-14_缓存命中率82%结构分析与日志时区修复.md` 追加"六、落地"小节: 方案、参数、模拟预测值。
- spec/plan 本身已在 docs/superpowers/ 下,无需额外索引。

## Task 7: 部署与验证（用户已授权）

```bash
cd /home/zhenyi/ownprojects/livetrans/frontend && bash sync-mini.sh -b
```

验证:
- `docker ps` 容器 Up;
- `docker logs livetrans --tail 5` 有 `Listening on http://[::]:8081`;
- `docker exec livetrans date` 为 CST;
- `docker exec livetrans env | grep -E 'NUXT_HISTORY|NUXT_PREVIEW'` 为空(mini .env 未覆盖 → 代码默认 20/100/3000 生效)。

## Task 8: tellme 通知

```bash
tellme "livetrans 缓存优化已部署: 锯齿窗口20/100+预览限流3s, 预计命中率82%→98%, 场均费用¥13→¥4.5。改动未提交,等你审核。"
```

## 停点

- git 提交: 完成后等待用户审核与提交指令(项目规范,禁止自动提交)。工作树中另有本任务之前产生的未提交改动(ai.ts ts 字段、docker-compose TZ、CLAUDE.md 索引), 提交时用户可拆分。
