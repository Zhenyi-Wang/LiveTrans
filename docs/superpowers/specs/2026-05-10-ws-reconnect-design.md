# WebSocket 断流断连重连设计

## Context

系统用于每周日礼拜直播的实时语音转文字。直播结束后流断开，系统应保持等待，下周日流恢复后自动继续转录。

当前问题：流断开后客户端保持 WebSocket 长连接，但服务端有 16.7 小时的 `max_connection_time` 超时。跨周等待必然超时，服务端主动断开 WebSocket 后，客户端 `recording` 被设为 `False`，之后即使流恢复、音频正常读取，也不会发送到服务端，转录无法恢复。同时 Ctrl+C 无法中断进程。

## 方案

断流时主动断开 WebSocket，流恢复时重建 WebSocket 连接。服务端不需要任何修改。

## 设计

### 改动范围

仅修改 `whisperlive/whisper_live/client.py`，不涉及服务端。

### 1. TranscriptionTeeClient 保存连接参数

当前 `Client` 实例创建后，连接参数（host、port、lang、model 等）没有保留。需要在 `TranscriptionTeeClient` 层保存这些参数，以便重连时复用。

保存内容：`host`、`port`、`lang`、`translate`、`model`、`use_vad`、`dispatch_api`、`srt_file_path`。

`TranscriptionClient.__init__` 中将已有参数传递给父类保存。

### 2. Client 类防御性修复

在 `Client.__init__` 中添加 `self.server_backend = None` 初始化，避免 `write_srt_file` 在 `SERVER_READY` 未到达时抛 `AttributeError`。

### 3. 新增 `disconnect_clients()` 方法

断流时调用：

1. `write_all_clients_srt()` — 写入当前场次 SRT
2. 遍历 clients，从 `Client.INSTANCES` 中移除旧实例（`Client.INSTANCES.pop(client.uid, None)`），防止内存泄漏
3. `close_all_clients()` — 关闭 WebSocket
4. 将 `self.clients` 置为空列表 `self.clients = []`（创建新列表，非原地修改）

### 4. 新增 `reconnect_clients()` 方法

流恢复时调用：

1. 用保存的连接参数创建新的 `Client` 实例
2. 等待服务端就绪，带超时机制（30 秒）：

```python
deadline = time.time() + 30
while not client.recording:
    if client.server_error:
        raise Exception("Server error during reconnection")
    if time.time() > deadline:
        raise Exception("Reconnection timeout: server not ready")
    time.sleep(0.1)
```

3. 赋值给 `self.clients = [client]`
4. 如果 `self` 是 `TranscriptionClient`（`hasattr(self, 'client')`），同步更新 `self.client = self.clients[0]`

### 5. close_websocket 超时保护

`Client.close_websocket()` 中 `ws_thread.join()` 改为 `ws_thread.join(timeout=5)`，超时后打印警告继续执行，避免网络异常时主线程死锁。

### 6. 修改 `handle_ffmpeg_process` 主循环

改造为三阶段：

```
while True:
    select + read ffmpeg

    if 读到空数据（流断开）:
        if clients 非空（WS 还连着）:
            disconnect_clients()
        stop + join consume_stderr 线程
        kill ffmpeg, sleep, 创建新 ffmpeg
        启动新 consume_stderr 线程
        continue

    if 读到音频数据:
        if clients 为空（WS 已断开）:
            reconnect_clients()
        bytes_to_float_array → multicast_packet
```

关键变化：
- 断流时立即断开 WS，不再发 PAUSE
- 流恢复时先建 WS 再发音频，不再发 RESET
- `reconnect_count` 仅用于日志，不再控制信号发送
- `send_pause_to_server` 和 `send_reset_to_server` 不再被调用
- `consume_stderr` 线程管理仍由 `handle_ffmpeg_process` 负责（断流时 stop+join，重建 ffmpeg 时重建线程）

### 7. Ctrl+C 可靠退出

当前 Ctrl+C 不可靠的原因：WS 断开后主循环无退出条件，快速循环中信号可能丢失。

修复：WS 断开后的等待循环中，每轮 `select` 超时后检查 `self.clients` 是否为空。如果 WS 断开且 ffmpeg 也无法重连，循环继续但 `select` 的 0.5s 超时确保信号有机会被处理。当前代码结构已有 `try/except KeyboardInterrupt`，配合 WS 断开重连后不再空转，Ctrl+C 问题间接解决。

### 8. 不改动的部分

- **服务端** `server.py`：新连接自然创建新 `ServeClientBase` 实例，`speech_to_text` daemon 线程通过 `exit=True` 自行退出，新连接创建全新实例
- **Client 类回调**：`on_open`、`on_message`、`on_close` 逻辑不变
- **`send_pause_to_server` / `send_reset_to_server`**：保留方法定义，不再调用，后续可清理

## 验证

1. 启动服务端和客户端，确认正常转录
2. 模拟流断开：停止 ffmpeg 源 → 观察日志输出 `disconnect_clients` → 确认 WS 关闭
3. 模拟流恢复：重启 ffmpeg 源 → 观察日志输出 `reconnect_clients` → 确认转录恢复
4. Ctrl+C 能正常退出
5. 不重启生产服务，仅修改代码文件
