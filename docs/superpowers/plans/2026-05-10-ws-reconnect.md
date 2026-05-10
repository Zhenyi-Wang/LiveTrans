# WebSocket 断流断连重连 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将客户端改为断流时断开 WebSocket、流恢复时重建连接，解决跨周等待超时导致转录无法恢复的问题。

**Architecture:** 仅修改 `client.py`。`TranscriptionTeeClient` 新增 `disconnect_clients()` 和 `reconnect_clients()` 方法，`handle_ffmpeg_process` 主循环改为三阶段（等待流→活跃转录→断流断WS）。

**Tech Stack:** Python 3, websocket-client, unittest.mock

**Spec:** `docs/superpowers/specs/2026-05-10-ws-reconnect-design.md`

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `whisperlive/whisper_live/client.py` | 修改 | 所有改动集中在此文件 |
| `whisperlive/tests/test_client.py` | 修改 | 新增 disconnect/reconnect 相关测试 |

---

### Task 1: Client 类防御性修复

**Files:**
- Modify: `whisperlive/whisper_live/client.py:52` (Client.__init__)
- Modify: `whisperlive/whisper_live/client.py:271-277` (close_websocket)

- [ ] **Step 1: 在 Client.__init__ 中初始化 server_backend**

在 `client.py` 第 52 行 `self.recording = False` 之后添加：

```python
self.server_backend = None
```

- [ ] **Step 2: close_websocket 添加 join 超时**

将 `client.py` 中 `close_websocket` 方法的 `self.ws_thread.join()` 改为 `self.ws_thread.join(timeout=5)`，并在超时后打印警告：

当前代码（约第 271-277 行）：
```python
def close_websocket(self):
    try:
        self.client_socket.close()
    except Exception as e:
        print("[ERROR]: Error closing WebSocket:", e)

    try:
        self.ws_thread.join()
    except Exception as e:
        print("[ERROR]: Error joining WebSocket thread:", e)
```

改为：
```python
def close_websocket(self):
    try:
        self.client_socket.close()
    except Exception as e:
        print("[ERROR]: Error closing WebSocket:", e)

    try:
        self.ws_thread.join(timeout=5)
        if self.ws_thread.is_alive():
            print("[WARN]: WebSocket thread did not exit within 5s timeout")
    except Exception as e:
        print("[ERROR]: Error joining WebSocket thread:", e)
```

- [ ] **Step 3: 运行现有测试确认不破坏**

Run: `cd /home/zhenyi/ownprojects/livetrans/whisperlive && python -m pytest tests/test_client.py -v`
Expected: 所有现有测试 PASS

---

### Task 2: TranscriptionTeeClient 保存连接参数

**Files:**
- Modify: `whisperlive/whisper_live/client.py:327-345` (TranscriptionTeeClient.__init__)
- Modify: `whisperlive/whisper_live/client.py:822-855` (TranscriptionClient.__init__)

- [ ] **Step 1: TranscriptionTeeClient.__init__ 增加连接参数存储**

当前代码（约第 327-345 行）：
```python
def __init__(
    self,
    clients,
    save_output_recording=False,
    output_recording_filename="./output_recording.wav",
):
    self.clients = clients
    if not self.clients:
        raise Exception("At least one client is required.")
    self.start_time = time.time()
    self.chunk = 4096
    ...
```

在 `self.clients = clients` 之后、`if not self.clients:` 之前添加：
```python
    # 保存连接参数，用于断流重连时重建 WebSocket
    self._client_params = {}
```

- [ ] **Step 2: TranscriptionClient.__init__ 保存连接参数**

当前代码（约第 822-855 行）：
```python
def __init__(
    self,
    host,
    port,
    lang=None,
    translate=False,
    model="small",
    use_vad=True,
    save_output_recording=False,
    output_recording_filename="./output_recording.wav",
    output_transcription_path="./output.srt",
    dispatch_api=None,
):
    self.client = Client(
        host,
        port,
        lang,
        translate,
        model,
        srt_file_path=output_transcription_path,
        use_vad=use_vad,
        dispatch_api=dispatch_api,
    )
    ...
    TranscriptionTeeClient.__init__(
        self,
```

在 `TranscriptionTeeClient.__init__` 调用之后添加：
```python
    # 保存连接参数用于断流后重连
    self._client_params = {
        "host": host,
        "port": port,
        "lang": lang,
        "translate": translate,
        "model": model,
        "use_vad": use_vad,
        "srt_file_path": output_transcription_path,
        "dispatch_api": dispatch_api,
    }
```

- [ ] **Step 3: 运行现有测试确认不破坏**

Run: `cd /home/zhenyi/ownprojects/livetrans/whisperlive && python -m pytest tests/test_client.py -v`
Expected: 所有现有测试 PASS

---

### Task 3: 新增 disconnect_clients 和 reconnect_clients 方法

**Files:**
- Modify: `whisperlive/whisper_live/client.py` (TranscriptionTeeClient 类，在 close_all_clients 之后)
- Modify: `whisperlive/tests/test_client.py` (新增测试)

- [ ] **Step 1: 写 disconnect_clients 的测试**

在 `test_client.py` 的 `TestTee` 类中添加：

```python
def test_disconnect_clients_clears_instances(self):
    uid1 = self.client2.uid
    uid2 = self.client3.uid
    self.assertIn(uid1, Client.INSTANCES)
    self.assertIn(uid2, Client.INSTANCES)
    self.tee.disconnect_clients()
    self.assertEqual(self.tee.clients, [])
    self.assertNotIn(uid1, Client.INSTANCES)
    self.assertNotIn(uid2, Client.INSTANCES)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/zhenyi/ownprojects/livetrans/whisperlive && python -m pytest tests/test_client.py::TestTee::test_disconnect_clients_clears_instances -v`
Expected: FAIL (AttributeError: 'TranscriptionTeeClient' has no attribute 'disconnect_clients')

- [ ] **Step 3: 实现 disconnect_clients**

在 `TranscriptionTeeClient` 类中，`close_all_clients` 方法之后添加：

```python
def disconnect_clients(self):
    """断流时断开所有 WebSocket 连接，清理 Client 实例。"""
    self.write_all_clients_srt()
    for client in self.clients:
        Client.INSTANCES.pop(client.uid, None)
    self.close_all_clients()
    self.clients = []
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/zhenyi/ownprojects/livetrans/whisperlive && python -m pytest tests/test_client.py::TestTee::test_disconnect_clients_clears_instances -v`
Expected: PASS

- [ ] **Step 5: 写 reconnect_clients 的测试**

在 `test_client.py` 的 `TestTee` 类中添加：

```python
@patch('whisper_live.client.websocket.WebSocketApp')
@patch('whisper_live.client.pyaudio.PyAudio')
def test_reconnect_clients_creates_new_client(self, mock_pyaudio, mock_ws):
    mock_pyaudio_inst = MagicMock()
    mock_pyaudio.return_value = mock_pyaudio_inst
    mock_stream = MagicMock()
    mock_pyaudio_inst.open.return_value = mock_stream

    self.tee.disconnect_clients()
    self.assertEqual(self.tee.clients, [])

    # 设置重连参数
    self.tee._client_params = {
        "host": "localhost",
        "port": 9090,
        "lang": "zh",
        "translate": False,
        "model": "small",
        "use_vad": True,
        "srt_file_path": "output.srt",
        "dispatch_api": None,
    }

    # mock Client.__init__，设置 recording=True 以跳过等待循环
    def mock_client_init(self, *a, **kw):
        self.recording = True
        self.server_error = False
        self.uid = "test-reconnect-uid"

    with patch.object(Client, '__init__', mock_client_init):
        self.tee.reconnect_clients()
        self.assertEqual(len(self.tee.clients), 1)
        self.assertEqual(self.tee.clients[0].uid, "test-reconnect-uid")
```

注意：mock_client_init 必须设置 `self.recording = True`，否则 reconnect_clients 内部的 `while not client.recording` 循环会永远阻塞。

- [ ] **Step 6: 实现 reconnect_clients**

在 `disconnect_clients` 方法之后添加：

```python
def reconnect_clients(self):
    """流恢复时重建 WebSocket 连接。"""
    if not self._client_params:
        raise Exception("No client params saved for reconnection")

    p = self._client_params
    client = Client(
        p["host"],
        p["port"],
        p.get("lang"),
        p.get("translate", False),
        p.get("model", "small"),
        srt_file_path=p.get("srt_file_path", "output.srt"),
        use_vad=p.get("use_vad", True),
        dispatch_api=p.get("dispatch_api"),
    )

    # 等待服务端就绪（SERVER_READY），带超时
    deadline = time.time() + 30
    while not client.recording:
        if client.server_error:
            raise Exception("Server error during reconnection")
        if time.time() > deadline:
            client.close_websocket()
            raise Exception("Reconnection timeout: server not ready within 30s")
        time.sleep(0.1)

    self.clients = [client]

    # 如果是 TranscriptionClient，同步更新 self.client 引用
    if hasattr(self, 'client'):
        self.client = client

    print(f"[INFO]: WebSocket reconnected, client uid={client.uid}")
```

- [ ] **Step 7: 运行所有测试**

Run: `cd /home/zhenyi/ownprojects/livetrans/whisperlive && python -m pytest tests/test_client.py -v`
Expected: 所有测试 PASS

---

### Task 4: 重写 handle_ffmpeg_process 主循环

**Files:**
- Modify: `whisperlive/whisper_live/client.py:504-597` (handle_ffmpeg_process)

- [ ] **Step 1: 重写 handle_ffmpeg_process**

将整个 `handle_ffmpeg_process` 方法替换为以下实现：

```python
def handle_ffmpeg_process(self, process, stream_type, create_process_func):
    print(f"[INFO]: Connecting to {stream_type} stream...")
    self.stop_stderr.clear()
    self.stderr_thread = threading.Thread(target=self.consume_stderr, args=(process,))
    self.stderr_thread.start()

    retry_delay = 2  # 初始延迟
    max_delay = 60   # 最大延迟
    first_disconnect_time = None  # 第一次断开的时间
    grace_period = 5400  # 90分钟内保持2秒间隔
    reconnect_count = 0  # 重连次数计数

    try:
        while True:
            # 用 select 轮询，避免阻塞 read() 导致无法响应 Ctrl+C
            while True:
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if ready:
                    break
            in_bytes = process.stdout.read(self.chunk * 2)  # 2 bytes per sample

            if not in_bytes:
                # 流断开，尝试重连
                now = time.time()
                reconnect_count += 1

                # 记录第一次断开时间
                if first_disconnect_time is None:
                    first_disconnect_time = now
                    reconnect_count = 1
                    print(f"[WARN] {stream_type} stream disconnected (first time), reconnect_count={reconnect_count}")
                    # 断开 WebSocket 连接
                    if self.clients:
                        print(f"[INFO] Disconnecting WebSocket due to stream loss...")
                        self.disconnect_clients()

                # 90分钟后开始指数退避
                if now - first_disconnect_time > grace_period:
                    retry_delay = min(retry_delay * 2, max_delay)

                print(f"[WARN] {stream_type} stream disconnected, retrying in {retry_delay}s... (reconnect #{reconnect_count}, disconnected_since={now-first_disconnect_time:.1f}s)")

                # 停止旧的 stderr 线程
                self.stop_stderr.set()
                if self.stderr_thread:
                    self.stderr_thread.join(timeout=1)
                self.stop_stderr.clear()

                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                time.sleep(retry_delay)

                # 重新创建 ffmpeg 进程
                process = create_process_func()
                self.stderr_thread = threading.Thread(target=self.consume_stderr, args=(process,))
                self.stderr_thread.start()
                continue

            # 流有数据
            if reconnect_count > 0:
                # 流刚恢复，重建 WebSocket 连接
                print(f"[INFO] {stream_type} stream reconnected after {reconnect_count} retries, rebuilding WebSocket...")
                retry_delay = 2
                first_disconnect_time = None
                reconnect_count = 0
                self.reconnect_clients()
                print(f"[INFO] WebSocket ready, resuming audio transmission")

            audio_array = self.bytes_to_float_array(in_bytes)
            self.multicast_packet(audio_array.tobytes())

    except KeyboardInterrupt:
        print(f"\n[INFO] Ctrl+C received, stopping {stream_type} stream...")
    except Exception as e:
        print(f"[ERROR]: Failed to connect to {stream_type} stream: {e}")
    finally:
        self.stop_stderr.set()
        if self.stderr_thread:
            self.stderr_thread.join(timeout=2)
        if self.clients:
            self.write_all_clients_srt()
            self.close_all_clients()
        if process:
            process.kill()

    print(f"[INFO]: {stream_type} stream processing finished.")
```

- [ ] **Step 2: 运行所有测试确认不破坏**

Run: `cd /home/zhenyi/ownprojects/livetrans/whisperlive && python -m pytest tests/test_client.py -v`
Expected: 所有测试 PASS

**设计决策：reconnect_clients 失败时的行为**

当 `reconnect_clients()` 抛出异常（服务端不可用或超时），异常被外层 `except Exception` 捕获，`handle_ffmpeg_process` 退出。这是预期行为——服务端不可用时需要人工介入，由 main.py 的进程管理机制重启整个服务。

---

### Task 5: 清理不再使用的 PAUSE/RESET 信号发送代码

**Files:**
- Modify: `whisperlive/whisper_live/client.py` (handle_ffmpeg_process 中已不调用)

- [ ] **Step 1: 确认 send_pause_to_server 和 send_reset_to_server 无其他调用者**

Run: `grep -rn "send_pause_to_server\|send_reset_to_server" /home/zhenyi/ownprojects/livetrans/whisperlive/ --include="*.py" | grep -v "def send_"`
Expected: 无结果（只有方法定义，无调用）

- [ ] **Step 2: 保留方法不删除**

这两个方法在 server.py 中仍有对应的处理逻辑。保留方法定义但不再调用，标记为 deprecated：

在 `send_pause_to_server` 方法前加注释：
```python
# DEPRECATED: 不再使用。断流时改为断开 WebSocket。
```

在 `send_reset_to_server` 方法前加注释：
```python
# DEPRECATED: 不再使用。流恢复时改为重建 WebSocket。
```

---

## Plan 自检

### Spec 覆盖

| Spec 要求 | 对应 Task |
|-----------|-----------|
| Client.__init__ 添加 server_backend=None | Task 1 Step 1 |
| close_websocket join 超时 | Task 1 Step 2 |
| TranscriptionTeeClient 保存连接参数 | Task 2 |
| disconnect_clients | Task 3 |
| reconnect_clients（含超时、server_error 检查、self.client 同步） | Task 3 |
| handle_ffmpeg_process 三阶段循环 | Task 4 |
| consume_stderr 线程管理 | Task 4（包含在循环中） |
| 清理 PAUSE/RESET 调用 | Task 5 |
| Ctrl+C 可靠退出 | Task 4（select 0.5s 超时 + 正常的 KeyboardInterrupt） |

### 占位符扫描

无 TBD、TODO、"implement later"、"add validation" 等占位符。

### 类型一致性

- `disconnect_clients()` 设置 `self.clients = []` — 与 `reconnect_clients()` 设置 `self.clients = [client]` 一致
- `Client.INSTANCES` pop 使用 `client.uid` — 与 `Client.__init__` 中 `Client.INSTANCES[self.uid] = self` 一致
- `_client_params` 字典的 key 与 `Client.__init__` 参数名一致
