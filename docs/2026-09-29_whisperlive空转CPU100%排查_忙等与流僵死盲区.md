# whisperlive 空转 CPU 100% 排查：忙等与流僵死盲区

> 日期：2026-09-29 · 定稿于当日
> 现象：非主日无任务时段，`run_server.py`（whisperlive 9090）单核 100%，持续 23 小时（累计 CPU 时间 1386 分钟）

## 现象与定位

- `ps` 显示 `python whisperlive/run_server.py --port 9090` 占满一核；`top -H` 定位到其内部单线程（非主线程）
- tmux `livetrans` 会话自启动（前日 18:19:47）后**零输出**——无报错、无重连日志
- ffmpeg 拉流进程（`http://mini:8080/live/livestream.flv`）存活但 23h 仅读 ~191KB、CPU 13 秒 = 连接僵死
- `ss` 显示 9090 的 WebSocket 一直 ESTABLISHED

## 根因（两层叠加）

### 根因 1：server 端转录线程忙等（直接烧 CPU 的位置）

`whisper_live/server.py` `ServeClientFasterWhisper.speech_to_text` 循环中，`frames_np is None`（客户端已连 WS 但从未收到音频帧）分支是**裸 `continue`**——无 sleep、无日志，静默烧满单核。

该分支源自本地魔改的"新算法"重写；上游原版（同文件 TensorRT 分支 `server.py` 的另一处）此路径本有 `time.sleep(0.02)`，重写时丢失。

**触发条件**：客户端已连接但无音频 → 常驻设计下周二至周六天天满足，只是一直没人注意 CPU 时间累计。

**修复**：该分支补 `time.sleep(0.05)`。实测稳态 CPU 从 100% → 0.0%。

### 根因 2：client 端"流僵死"检测盲区（让根因 1 长期成立的原因）

`whisper_live/client.py` `handle_ffmpeg_process` 的断流检测**只认 ffmpeg stdout EOF**（`not in_bytes`）。退避重连（2s→60s）、90 分钟停 server 等机制全部挂在这个检测之后。

但存在第三种状态：**对端连接挂起不响应**——ffmpeg 进程活着、不退出、stdout 永不返回 EOF 也无数据。此时：

- 断流事件永不触发 → 退避/重连/停 server 全部失效（日志一行都没有）
- WS 保持连接 → server 端认为"客户端在、等音频" → 根因 1 的忙等被无限供养

ffmpeg cmdline 未设 `-rw_timeout` 等超时，对端不响应时无限干等。curl 实测 mini:8080 当时不可达（连接挂起而非快速拒绝），正好落入此盲区。

**修复**：读循环 select 等待处跟踪 `last_data_time`，超过 `no_data_timeout=60s` 无任何数据 → 打 `[STREAM] ... no data for 60s (ffmpeg hung), forcing reconnect`，置 `in_bytes=b""` 模拟 EOF，**复用既有断流重连路径**（不另建平行逻辑）。`last_data_time` 在三处重置：每次成功 read 后、重连路径 recreate ffmpeg 后、暂停恢复 recreate ffmpeg 后（暂停等待可达数小时，不重置会恢复即误判）。

## 实测（2026-09-29 晚，mini 无流 = 天然僵死场景）

```
19:08:43 [PAUSE] 已恢复 (ffmpeg pid=894968)          ← resume，ffmpeg 起来即僵死
19:09:43 [STREAM] Other stream no data for 60s (ffmpeg hung), forcing reconnect
19:09:43 [STREAM] Other stream disconnected (first time)
19:09:43 [STREAM] Reconnect #1, retry in 2s
19:10:46 [STREAM] Other stream no data for 60s (ffmpeg hung), forcing reconnect
19:10:46 [STREAM] Reconnect #2, retry in 2s (offline 63s, server=running)
```

- 超时精确 60s 触发，两轮循环稳定，offline 时长统计正确
- server 进程稳态 CPU 0.0%（对照修复前同场景 100%）
- 与暂停功能交互正常（pause/resume 往返、teardown/重建、`last_data_time` 重置生效）

## 经验教训

1. **常驻服务的"无输入"路径必须显式让出 CPU**：任何 `while True + continue` 分支都要有 sleep，哪怕业务上"不该停留很久"
2. **断流检测不能只认 EOF**：网络故障三态（快速失败/快速 EOF/挂起不响应）里挂起态最阴险——进程活着、无错误、无日志。读超时兜底（无数据 N 秒视为断）是必须的
3. **全静默的故障路径最难发现**：本次忙等恰好是全循环唯一不 sleep 不 print 的分支；给等待分支留一行周期性日志能大幅降低定位成本
4. 排查入口：`ps` 看 CPU 时间累计 vs 启动时长，`top -H` 定位线程，`/proc/<pid>/io` 看 ffmpeg 是否真在读
