import json
import os
import select
import shutil
import signal
import subprocess
import threading
import time
import uuid
import wave

import ffmpeg
import logging
import numpy as np
import pyaudio
import requests
import websocket

import whisper_live.utils as utils


class Client:
    """
    Handles communication with a server using WebSocket.
    """

    INSTANCES = {}
    END_OF_AUDIO = "END_OF_AUDIO"

    def __init__(
        self,
        host=None,
        port=None,
        lang=None,
        translate=False,
        model="small",
        srt_file_path="output.srt",
        use_vad=True,
        dispatch_api=None,
    ):
        """
        Initializes a Client instance for audio recording and streaming to a server.

        If host and port are not provided, the WebSocket connection will not be established.
        When translate is True, the task will be set to "translate" instead of "transcribe".
        he audio recording starts immediately upon initialization.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number for the WebSocket server.
            lang (str, optional): The selected language for transcription. Default is None.
            translate (bool, optional): Specifies if the task is translation. Default is False.
        """
        self.recording = False
        self.server_backend = None
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.language = lang
        self.model = model
        self.server_error = False
        self.srt_file_path = srt_file_path
        self.use_vad = use_vad
        self.last_segment = None
        self.last_received_segment = None
        self.dispatch_api = dispatch_api

        if translate:
            self.task = "translate"

        self.timestamp_offset = 0.0
        self.audio_bytes = None

        # dispatch 异步队列:WS回调只入队,HTTP发送在独立线程重试,前端不可用时
        # 不阻塞WS收发(Broken pipe断流的根因),恢复后自动续传
        if dispatch_api:
            import queue
            self.dispatch_queue = queue.Queue(maxsize=500)
            self.dispatch_thread = threading.Thread(target=self._dispatch_worker, daemon=True)
            self.dispatch_thread.start()

        if host is not None and port is not None:
            socket_url = f"ws://{host}:{port}"
            self.client_socket = websocket.WebSocketApp(
                socket_url,
                on_open=lambda ws: self.on_open(ws),
                on_message=lambda ws, message: self.on_message(ws, message),
                on_error=lambda ws, error: self.on_error(ws, error),
                on_close=lambda ws, close_status_code, close_msg: self.on_close(
                    ws, close_status_code, close_msg
                ),
            )
        else:
            print("[ERROR]: No host or port specified.")
            return

        Client.INSTANCES[self.uid] = self

        # start websocket client in a thread
        self.ws_thread = threading.Thread(target=self.client_socket.run_forever)
        self.ws_thread.setDaemon(True)
        self.ws_thread.start()

        self.transcript = []
        print("[INFO]: * recording")

    def handle_status_messages(self, message_data):
        """Handles server status messages."""
        status = message_data["status"]
        if status == "WAIT":
            self.waiting = True
            print(
                f"[INFO]: Server is full. Estimated wait time {round(message_data['message'])} minutes."
            )
        elif status == "ERROR":
            print(f"Message from Server: {message_data['message']}")
            self.server_error = True
        elif status == "WARNING":
            print(f"Message from Server: {message_data['message']}")

    def process_segments(self, segments):
        """Processes transcript segments."""
        # text = []
        # for i, seg in enumerate(segments):
        #     if not text or text[-1] != seg["text"]:
        #         text.append(seg["text"])
        #         if i == len(segments) - 1:
        #             self.last_segment = seg
        #         elif (self.server_backend == "faster_whisper" and
        #               (not self.transcript or
        #                 float(seg['start']) >= float(self.transcript[-1]['end']))):
        #             self.transcript.append(seg)
        # # update last received segment and last valid response time
        # if self.last_received_segment is None or self.last_received_segment != segments[-1]["text"]:
        #     self.last_response_received = time.time()
        #     self.last_received_segment = segments[-1]["text"]

        # Truncate to last 3 entries for brevity.
        # text = text[-3:]
        # utils.clear_screen()
        # utils.print_transcript(text)
        # print("-" * 30)
        # print('='*30)
        # for t in self.transcript:
        #     print(t)

        print("Client sending segments to dispatch API", segments)

        if self.dispatch_api:
            segments["_enqueued_at"] = time.time()
            try:
                self.dispatch_queue.put_nowait(segments)
            except Exception:
                # 队列满(前端长时间不可用):丢弃最旧一条,保最新
                try:
                    self.dispatch_queue.get_nowait()
                    self.dispatch_queue.put_nowait(segments)
                except Exception:
                    pass

    def _dispatch_worker(self):
        """独立线程消费dispatch队列,失败退避重试直到成功"""
        while True:
            segments = self.dispatch_queue.get()
            # 积压重放提速:过旧的纯current预览直接丢弃(confirmed永不丢)
            if (not segments.get("confirmed")
                    and time.time() - segments.get("_enqueued_at", 0) > 15):
                continue
            while True:
                try:
                    requests.post(self.dispatch_api, data=json.dumps(segments), timeout=3)
                    break
                except Exception as e:
                    print(f"[DISPATCH] 发送失败,5s后重试: {e}")
                    time.sleep(5)

    def on_message(self, ws, message):
        """
        Callback function called when a message is received from the server.

        It updates various attributes of the client based on the received message, including
        recording status, language detection, and server messages. If a disconnect message
        is received, it sets the recording status to False.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.
            message (str): The received message from the server.

        """
        message = json.loads(message)

        if self.uid != message.get("uid"):
            print(f"[{self.ts()}] [ERROR] invalid client uid")
            return

        if "status" in message.keys():
            self.handle_status_messages(message)
            return

        if "message" in message.keys() and message["message"] == "DISCONNECT":
            print(f"[{self.ts()}] [WARN] Server disconnected due to overtime")
            self.recording = False

        if "message" in message.keys() and message["message"] == "SERVER_READY":
            self.last_response_received = time.time()
            self.recording = True
            self.server_backend = message["backend"]
            print(f"[{self.ts()}] [WS] SERVER_READY (backend={self.server_backend}, uid={self.uid})")
            return

        if "language" in message.keys():
            self.language = message.get("language")
            lang_prob = message.get("language_prob")
            print(f"[{self.ts()}] [WS] Language detected: {self.language} (prob={lang_prob})")
            return

        if "segments" in message.keys():
            self.process_segments(message["segments"])

    def on_error(self, ws, error):
        print(f"[{self.ts()}] [WS] Error: {error}")
        self.server_error = True
        self.error_message = error

    def on_close(self, ws, close_status_code, close_msg):
        print(f"[{self.ts()}] [WS] Connection closed: code={close_status_code}, msg={close_msg}")
        self.recording = False
        self.waiting = False

    @staticmethod
    def ts():
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def on_open(self, ws):
        """
        Callback function called when the WebSocket connection is successfully opened.

        Sends an initial configuration message to the server, including client UID,
        language selection, and task type.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.

        """
        print(f"[{self.ts()}] [WS] Connection opened, sending config (uid={self.uid}, lang={self.language}, model={self.model})")
        ws.send(
            json.dumps(
                {
                    "uid": self.uid,
                    "language": self.language,
                    "task": self.task,
                    "model": self.model,
                    "use_vad": self.use_vad,
                }
            )
        )

    def send_packet_to_server(self, message):
        """
        Send an audio packet to the server using WebSocket.

        Args:
            message (bytes): The audio data packet in bytes to be sent to the server.

        """
        try:
            self.client_socket.send(message, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            print(e)

    # DEPRECATED: 不再使用。断流时改为断开 WebSocket。
    def send_reset_to_server(self):
        """
        Send a reset signal to the server to clear audio buffer state.
        Called when stream reconnects after disconnection.
        """
        try:
            reset_message = json.dumps({"type": "RESET_AUDIO_BUFFER"})
            self.client_socket.send(reset_message)
            print("[DEBUG] Sent RESET_AUDIO_BUFFER to server")
        except Exception as e:
            print(f"[ERROR] Failed to send reset signal: {e}")

    # DEPRECATED: 不再使用。流恢复时改为重建 WebSocket。
    def send_pause_to_server(self):
        """
        Send a pause signal to the server to pause processing.
        Called when stream disconnects.
        """
        try:
            pause_message = json.dumps({"type": "PAUSE"})
            self.client_socket.send(pause_message)
            print("[DEBUG] Sent PAUSE to server")
        except Exception as e:
            print(f"[ERROR] Failed to send pause signal: {e}")

    def close_websocket(self):
        """
        Close the WebSocket connection and join the WebSocket thread.

        First attempts to close the WebSocket connection using `self.client_socket.close()`. After
        closing the connection, it joins the WebSocket thread to ensure proper termination.

        """
        try:
            self.client_socket.close()
        except Exception as e:
            print("[ERROR]: Error closing WebSocket:", e)

        try:
            self.ws_thread.join(timeout=5)
            if self.ws_thread.is_alive():
                print("[WARN]: WebSocket thread did not exit within 5s timeout")
        except Exception as e:
            print("[ERROR:] Error joining WebSocket thread:", e)

    def get_client_socket(self):
        """
        Get the WebSocket client socket instance.

        Returns:
            WebSocketApp: The WebSocket client socket instance currently in use by the client.
        """
        return self.client_socket

    def write_srt_file(self, output_path="output.srt"):
        """
        Writes out the transcript in .srt format.

        Args:
            message (output_path, optional): The path to the target file.  Default is "output.srt".

        """
        if self.server_backend == "faster_whisper":
            if self.last_segment:
                self.transcript.append(self.last_segment)
            utils.create_srt_file(self.transcript, output_path)

    def wait_before_disconnect(self):
        """Waits a bit before disconnecting in order to process pending responses."""
        assert self.last_response_received
        while (
            time.time() - self.last_response_received
            < self.disconnect_if_no_response_for
        ):
            continue


class TranscriptionTeeClient:
    """
    Client for handling audio recording, streaming, and transcription tasks via one or more
    WebSocket connections.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to one or more servers, and receive transcribed text segments.
    Args:
        clients (list): one or more previously initialized Client instances

    Attributes:
        clients (list): the underlying Client instances responsible for handling WebSocket connections.
    """

    def __init__(
        self,
        clients,
        save_output_recording=False,
        output_recording_filename="./output_recording.wav",
    ):
        self.clients = clients
        self._client_params = {}
        self._server_command = None
        self._server_process = None
        if not self.clients:
            raise Exception("At least one client is required.")
        self.start_time = time.time()  # 追踪客户端开始运行时间
        self.chunk = 4096
        # self.chunk = 16000
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 30000
        self.save_output_recording = save_output_recording
        self.output_recording_filename = output_recording_filename
        self.frames = b""
        self.p = pyaudio.PyAudio()
        self.stderr_thread = None
        self.stop_stderr = threading.Event()
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
        except OSError as error:
            print(f"[WARN]: Unable to access microphone. {error}")
            self.stream = None

    def __call__(self, audio=None, rtsp_url=None, hls_url=None, other_url=None, save_file=None):
        """
        Start the transcription process.

        Initiates the transcription process by connecting to the server via a WebSocket. It waits for the server
        to be ready to receive audio data and then sends audio for transcription. If an audio file is provided, it
        will be played and streamed to the server; otherwise, it will perform live recording.

        Args:
            audio (str, optional): Path to an audio file for transcription. Default is None, which triggers live recording.

        """
        assert (
            sum(source is not None for source in [audio, rtsp_url, hls_url]) <= 1
        ), "You must provide only one selected source"

        print(f"[{Client.ts()}] [INIT] Waiting for server ready...")
        attempt = 0
        for client in self.clients:
            deadline = time.time() + 120  # 最多等2分钟
            while not client.recording:
                if client.waiting or client.server_error:
                    if self._server_command and time.time() < deadline:
                        attempt += 1
                        print(f"[{Client.ts()}] [INIT] Server not ready (attempt #{attempt}), retrying in 2s...")
                        Client.INSTANCES.pop(client.uid, None)
                        client.close_websocket()
                        p = self._client_params
                        client = Client(
                            p["host"], p["port"], p.get("lang"), p.get("translate", False),
                            p.get("model", "small"), srt_file_path=p.get("srt_file_path", "output.srt"),
                            use_vad=p.get("use_vad", True), dispatch_api=p.get("dispatch_api"),
                        )
                        self.clients = [client]
                        if hasattr(self, 'client'):
                            self.client = client
                        time.sleep(5)
                        continue
                    print(f"[{Client.ts()}] [INIT] Server failed to become ready within 120s, giving up")
                    self.close_all_clients()
                    return

        print(f"[{Client.ts()}] [INIT] Server ready! Starting stream processing...")
        if hls_url is not None:
            self.process_hls_stream(hls_url, save_file)
        elif audio is not None:
            resampled_file = utils.resample(audio)
            self.play_file(resampled_file)
        elif rtsp_url is not None:
            self.process_rtsp_stream(rtsp_url)
        elif other_url is not None:
            self.process_other_stream(other_url)
        else:
            self.record()

    def close_all_clients(self):
        """Closes all client websockets."""
        for client in self.clients:
            client.close_websocket()

    def write_all_clients_srt(self):
        """Writes out .srt files for all clients."""
        for client in self.clients:
            client.write_srt_file(client.srt_file_path)

    def disconnect_clients(self):
        """断流时断开所有 WebSocket 连接，清理 Client 实例。"""
        self.write_all_clients_srt()
        for client in self.clients:
            Client.INSTANCES.pop(client.uid, None)
        self.close_all_clients()
        self.clients = []

    def reconnect_clients(self):
        """流恢复时重建 WebSocket 连接。"""
        if not self._client_params:
            raise Exception("No client params saved for reconnection")

        p = self._client_params
        print(f"[{Client.ts()}] [RECONNECT] Creating new WebSocket client (host={p['host']}:{p['port']}, model={p.get('model')})")
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

        deadline = time.time() + 60
        last_log = time.time()
        while not client.recording:
            if client.server_error:
                Client.INSTANCES.pop(client.uid, None)
                raise Exception("Server error during reconnection")
            if time.time() > deadline:
                client.close_websocket()
                Client.INSTANCES.pop(client.uid, None)
                raise Exception("Reconnection timeout: server not ready within 60s")
            if time.time() - last_log > 10:
                elapsed = time.time() - (deadline - 60)
                print(f"[{Client.ts()}] [RECONNECT] Waiting for SERVER_READY... ({elapsed:.0f}s/60s)")
                last_log = time.time()
            time.sleep(0.1)

        self.clients = [client]
        if hasattr(self, 'client'):
            self.client = client

        print(f"[{Client.ts()}] [RECONNECT] WebSocket connected (uid={client.uid})")

    def _kill_port_users(self, port):
        """杀掉占用指定端口的进程。"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("127.0.0.1", port))
                s.close()
            port_in_use = True
        except (ConnectionRefusedError, OSError):
            port_in_use = False

        if not port_in_use:
            return

        try:
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                pid = pid.strip()
                if not pid:
                    continue
                print(f"[{Client.ts()}] [SERVER] Killing stale process on port {port} (pid={pid})")
                os.kill(int(pid), signal.SIGTERM)
            if pids and pids[0]:
                time.sleep(1)
        except Exception as e:
            print(f"[{Client.ts()}] [SERVER] Failed to check port {port}: {e}")

    def _wait_for_port(self, port, timeout=120):
        """等待端口开始监听。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._server_process and self._server_process.poll() is not None:
                raise Exception(f"Server process exited with code {self._server_process.returncode}")
            try:
                result = subprocess.run(
                    ["ss", "-tlnp"],
                    capture_output=True, text=True, timeout=5,
                )
                if f":{port}" in result.stdout and "LISTEN" in result.stdout:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise Exception(f"Server did not start listening on port {port} within {timeout}s")

    _LOG_NOISE_PATTERNS = (
        "ALSA lib", "Cannot get card", "Cannot open device",
        "Unknown PCM", "Invalid card", "Invalid field",
        "pcm_oss", "pcm_usb", "snd_pcm", "snd_func",
        "snd_config", "_snd_pcm", "Evaluate error",
        "DEBUG:websockets", "ERROR:websockets",
        "[DEBUG add_frames]", "[DEBUG process_loop]",
        "[DEBUG SKIP]", "[DEBUG TRANSCRIBE]",
        "------------------------------",
        "Runtime:", "[DEBUG result]",
        "Saved last_trans_params", "Good, sleep",
        "duration:", "DEBUG:faster_whisper",
    )
    _NOISE_BLOCK_TRIGGERS = ("did not receive a valid HTTP request", "connection closed while reading HTTP request line")

    def _server_log_reader(self, proc, log_file):
        """后台线程：读取 server 输出，过滤噪音后 tee 到控制台和日志文件。"""
        in_noise_block = False
        try:
            for line in proc.stdout:
                line_str = line.decode("utf-8", errors="replace").rstrip()
                log_file.write(line_str + "\n")
                log_file.flush()

                # 检测并跳过已知的噪音 traceback 块
                if any(t in line_str for t in self._NOISE_BLOCK_TRIGGERS):
                    in_noise_block = True
                if in_noise_block:
                    if line_str == "" or line_str.startswith("  ") or line_str.startswith("Traceback") or line_str.startswith("The above"):
                        continue
                    in_noise_block = False

                if any(line_str.startswith(p) for p in self._LOG_NOISE_PATTERNS):
                    continue
                print(f"[SERVER] {line_str}")
        except Exception:
            pass

    def start_server(self):
        """启动 server 子进程。仅在 server_command 已配置且 server 未运行时启动。"""
        if not self._server_command:
            return
        if self._server_process and self._server_process.poll() is None:
            print(f"[{Client.ts()}] [SERVER] Already running (pid={self._server_process.pid})")
            return

        # 提取端口号并清理残留进程
        port = 9090
        for i, arg in enumerate(self._server_command):
            if arg == "--port" and i + 1 < len(self._server_command):
                try:
                    port = int(self._server_command[i + 1])
                except ValueError:
                    pass
        self._kill_port_users(port)

        server_log = open("server.log", "a")
        print(f"[{Client.ts()}] [SERVER] Starting: {' '.join(self._server_command)}")
        server_env = os.environ.copy()
        server_env["PULSE_SERVER"] = ""  # 抑制 ALSA 警告
        server_env["WEBLOG_LEVEL"] = "INFO"  # websockets 日志级别
        self._server_process = subprocess.Popen(
            self._server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=server_env,
        )
        threading.Thread(
            target=self._server_log_reader,
            args=(self._server_process, server_log),
            daemon=True,
        ).start()
        self._wait_for_port(port)
        print(f"[{Client.ts()}] [SERVER] Started and listening on port {port} (pid={self._server_process.pid})")

    def stop_server(self):
        """停止 server 子进程。"""
        if not self._server_process:
            return
        if self._server_process.poll() is not None:
            print(f"[{Client.ts()}] [SERVER] Already stopped (exit_code={self._server_process.returncode})")
            self._server_process = None
            return
        print(f"[{Client.ts()}] [SERVER] Stopping (pid={self._server_process.pid})...")
        self._server_process.terminate()
        try:
            self._server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"[{Client.ts()}] [SERVER] terminate timeout, killing...")
            self._server_process.kill()
            self._server_process.wait(timeout=3)
        print(f"[{Client.ts()}] [SERVER] Stopped (exit_code={self._server_process.returncode})")
        self._server_process = None

    def multicast_packet(self, packet, unconditional=False):
        """
        Sends an identical packet via all clients.

        Args:
            packet (bytes): The audio data packet in bytes to be sent.
            unconditional (bool, optional): If true, send regardless of whether clients are recording.  Default is False.
        """
        for client in self.clients:
            if unconditional or client.recording:
                client.send_packet_to_server(packet)

    def play_file(self, filename):
        """
        Play an audio file and send it to the server for processing.

        Reads an audio file, plays it through the audio output, and simultaneously sends
        the audio data to the server for processing. It uses PyAudio to create an audio
        stream for playback. The audio data is read from the file in chunks, converted to
        floating-point format, and sent to the server using WebSocket communication.
        This method is typically used when you want to process pre-recorded audio and send it
        to the server in real-time.

        Args:
            filename (str): The path to the audio file to be played and sent to the server.
        """

        # read audio and create pyaudio stream
        with wave.open(filename, "rb") as wavfile:
            self.stream = self.p.open(
                format=self.p.get_format_from_width(wavfile.getsampwidth()),
                channels=wavfile.getnchannels(),
                rate=wavfile.getframerate(),
                input=True,
                output=True,
                frames_per_buffer=self.chunk,
            )
            try:
                while any(client.recording for client in self.clients):
                    data = wavfile.readframes(self.chunk)
                    if data == b"":
                        break

                    audio_array = self.bytes_to_float_array(data)
                    self.multicast_packet(audio_array.tobytes())
                    self.stream.write(data)

                wavfile.close()

                for client in self.clients:
                    client.wait_before_disconnect()
                self.multicast_packet(Client.END_OF_AUDIO.encode("utf-8"), True)
                self.write_all_clients_srt()
                self.stream.close()
                self.close_all_clients()

            except KeyboardInterrupt:
                wavfile.close()
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                self.close_all_clients()
                self.write_all_clients_srt()
                print("[INFO]: Keyboard interrupt.")

    def process_rtsp_stream(self, rtsp_url):
        """
        Connect to an RTSP source, process the audio stream, and send it for trascription.

        Args:
            rtsp_url (str): The URL of the RTSP stream source.
        """
        process = self.get_rtsp_ffmpeg_process(rtsp_url)
        self.handle_ffmpeg_process(process, stream_type="RTSP", create_process_func=lambda: self.get_rtsp_ffmpeg_process(rtsp_url))

    def process_hls_stream(self, hls_url, save_file):
        """
        Connect to an HLS source, process the audio stream, and send it for transcription.

        Args:
            hls_url (str): The URL of the HLS stream source.
            save_file （str, optional): Local path to save the network stream.
        """
        process = self.get_hls_ffmpeg_process(hls_url, save_file)
        self.handle_ffmpeg_process(process, stream_type="HLS", create_process_func=lambda: self.get_hls_ffmpeg_process(hls_url, save_file))

    def process_other_stream(self, other_url):
        """
        Connect to an online source, process the audio stream, and send it for trascription.

        Args:
            other_url (str): The URL of the stream source.
        """
        process = self.get_rtsp_ffmpeg_process(other_url)
        self.handle_ffmpeg_process(process, stream_type="Other", create_process_func=lambda: self.get_rtsp_ffmpeg_process(other_url))


    def handle_ffmpeg_process(self, process, stream_type, create_process_func):
        print(f"[{Client.ts()}] [STREAM] Connecting to {stream_type} stream (ffmpeg pid={process.pid})...")
        self.stop_stderr.clear()
        self.stderr_thread = threading.Thread(target=self.consume_stderr, args=(process,))
        self.stderr_thread.start()

        retry_delay = 2  # 初始延迟
        max_delay = 60   # 最大延迟
        first_disconnect_time = None  # 第一次断开的时间
        grace_period = 5400  # 90分钟内保持2秒间隔
        server_idle_timeout = grace_period  # 断流90分钟后关闭server，与指数退避同步
        server_stopped = False  # 标记server是否已被关闭
        reconnect_count = 0  # 重连次数计数
        prev_retry_delay = retry_delay  # 用于检测退避变化
        audio_packet_count = 0  # 已发送的音频包计数

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
                        print(f"[{Client.ts()}] [STREAM] {stream_type} stream disconnected (first time)")
                        # 断开 WebSocket 连接
                        if self.clients:
                            self.disconnect_clients()

                    # 断流超过阈值，关闭server释放GPU
                    if not server_stopped and now - first_disconnect_time > server_idle_timeout:
                        print(f"[{Client.ts()}] [STREAM] Offline for {now - first_disconnect_time:.0f}s > {server_idle_timeout}s threshold, stopping server")
                        self.stop_server()
                        server_stopped = True

                    # 90分钟后开始指数退避
                    if now - first_disconnect_time > grace_period:
                        retry_delay = min(retry_delay * 2, max_delay)

                    if retry_delay != prev_retry_delay:
                        print(f"[{Client.ts()}] [STREAM] Retry delay changed: {prev_retry_delay}s → {retry_delay}s")
                        prev_retry_delay = retry_delay

                    if reconnect_count % 100 == 1 or reconnect_count <= 3:
                        print(f"[{Client.ts()}] [STREAM] Reconnect #{reconnect_count}, retry in {retry_delay}s (offline {now-first_disconnect_time:.0f}s, server={'stopped' if server_stopped else 'running'})")

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
                    print(f"[{Client.ts()}] [STREAM] {stream_type} stream recovered after {reconnect_count} retries, reconnecting...")
                    retry_delay = 2
                    prev_retry_delay = retry_delay
                    first_disconnect_time = None
                    reconnect_count = 0
                    if server_stopped:
                        self.start_server()
                        server_stopped = False
                    self.reconnect_clients()
                    print(f"[{Client.ts()}] [STREAM] Fully reconnected, resuming audio transmission")

                audio_array = self.bytes_to_float_array(in_bytes)
                self.multicast_packet(audio_array.tobytes())
                audio_packet_count += 1
                if audio_packet_count == 1:
                    print(f"[{Client.ts()}] [STREAM] First audio packet sent ({len(in_bytes)} bytes)")
                elif audio_packet_count % 1000 == 0:
                    print(f"[{Client.ts()}] [STREAM] Audio packets sent: {audio_packet_count}")

        except KeyboardInterrupt:
            print(f"\n[{Client.ts()}] [STREAM] Ctrl+C received")
        except Exception as e:
            print(f"[{Client.ts()}] [STREAM] Fatal error: {e}")
        finally:
            self.stop_stderr.set()
            if self.stderr_thread:
                self.stderr_thread.join(timeout=2)
            print(f"[{Client.ts()}] [STREAM] Cleaning up (audio_packets={audio_packet_count}, clients={len(self.clients)})")
            if self.clients:
                self.write_all_clients_srt()
                self.close_all_clients()
            self.stop_server()
            if process:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        print(f"[{Client.ts()}] [STREAM] {stream_type} processing finished (total_packets={audio_packet_count})")

    def get_rtsp_ffmpeg_process(self, rtsp_url):
        return (
            ffmpeg.input(rtsp_url, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.rate)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    def get_hls_ffmpeg_process(self, hls_url, save_file):
        if save_file is None:
            process = (
                ffmpeg.input(hls_url, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.rate)
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
        else:
            input = ffmpeg.input(hls_url, threads=0)
            output_file = input.output(
                save_file, acodec="copy", vcodec="copy"
            ).global_args("-loglevel", "quiet")
            output_std = input.output(
                "-", format="s16le", acodec="pcm_s16le", ac=1, ar=self.rate
            )
            process = ffmpeg.merge_outputs(output_file, output_std).run_async(
                pipe_stdout=True, pipe_stderr=True
            )

        return process

    def consume_stderr(self, process):
        """
        Consume and log the stderr output of a process in a separate thread.

        Args:
            process (subprocess.Popen): The process whose stderr output will be logged.
        """
        for line in iter(process.stderr.readline, b""):
            if self.stop_stderr.is_set():
                break
            decoded = line.decode().strip()
            if decoded:
                logging.debug(f'[STDERR]: {decoded}')

    def save_chunk(self, n_audio_file):
        """
        Saves the current audio frames to a WAV file in a separate thread.

        Args:
        n_audio_file (int): The index of the audio file which determines the filename.
                            This helps in maintaining the order and uniqueness of each chunk.
        """
        t = threading.Thread(
            target=self.write_audio_frames_to_file,
            args=(
                self.frames[:],
                f"chunks/{n_audio_file}.wav",
            ),
        )
        t.start()

    def finalize_recording(self, n_audio_file):
        """
        Finalizes the recording process by saving any remaining audio frames,
        closing the audio stream, and terminating the process.

        Args:
        n_audio_file (int): The file index to be used if there are remaining audio frames to be saved.
                            This index is incremented before use if the last chunk is saved.
        """
        if self.save_output_recording and len(self.frames):
            self.write_audio_frames_to_file(
                self.frames[:], f"chunks/{n_audio_file}.wav"
            )
            n_audio_file += 1
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.close_all_clients()
        if self.save_output_recording:
            self.write_output_recording(n_audio_file)
        self.write_all_clients_srt()

    def record(self):
        """
        Record audio data from the input stream and save it to a WAV file.

        Continuously records audio data from the input stream, sends it to the server via a WebSocket
        connection, and simultaneously saves it to multiple WAV files in chunks. It stops recording when
        the `RECORD_SECONDS` duration is reached or when the `RECORDING` flag is set to `False`.

        Audio data is saved in chunks to the "chunks" directory. Each chunk is saved as a separate WAV file.
        The recording will continue until the specified duration is reached or until the `RECORDING` flag is set to `False`.
        The recording process can be interrupted by sending a KeyboardInterrupt (e.g., pressing Ctrl+C). After recording,
        the method combines all the saved audio chunks into the specified `out_file`.
        """
        n_audio_file = 0
        if self.save_output_recording:
            if os.path.exists("chunks"):
                shutil.rmtree("chunks")
            os.makedirs("chunks")
        try:
            for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                if not any(client.recording for client in self.clients):
                    break
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames += data

                audio_array = self.bytes_to_float_array(data)

                self.multicast_packet(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > 60 * self.rate:
                    if self.save_output_recording:
                        self.save_chunk(n_audio_file)
                        n_audio_file += 1
                    self.frames = b""
            self.write_all_clients_srt()

        except KeyboardInterrupt:
            self.finalize_recording(n_audio_file)

    def write_audio_frames_to_file(self, frames, file_name):
        """
        Write audio frames to a WAV file.

        The WAV file is created or overwritten with the specified name. The audio frames should be
        in the correct format and match the specified channel, sample width, and sample rate.

        Args:
            frames (bytes): The audio frames to be written to the file.
            file_name (str): The name of the WAV file to which the frames will be written.

        """
        with wave.open(file_name, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    def write_output_recording(self, n_audio_file):
        """
        Combine and save recorded audio chunks into a single WAV file.

        The individual audio chunk files are expected to be located in the "chunks" directory. Reads each chunk
        file, appends its audio data to the final recording, and then deletes the chunk file. After combining
        and saving, the final recording is stored in the specified `out_file`.


        Args:
            n_audio_file (int): The number of audio chunk files to combine.
            out_file (str): The name of the output WAV file to save the final recording.

        """
        input_files = [
            f"chunks/{i}.wav"
            for i in range(n_audio_file)
            if os.path.exists(f"chunks/{i}.wav")
        ]
        with wave.open(self.output_recording_filename, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for in_file in input_files:
                with wave.open(in_file, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                # remove this file
                os.remove(in_file)
        wavfile.close()
        # clean up temporary directory to store chunks
        if os.path.exists("chunks"):
            shutil.rmtree("chunks")

    @staticmethod
    def bytes_to_float_array(audio_bytes):
        """
        Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data is normalized to
        have values between -1 and 1.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


class TranscriptionClient(TranscriptionTeeClient):
    """
    Client for handling audio transcription tasks via a single WebSocket connection.

    Acts as a high-level client for audio transcription tasks using a WebSocket connection. It can be used
    to send audio data for transcription to a server and receive transcribed text segments.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number to connect to on the server.
        lang (str, optional): The primary language for transcription. Default is None, which defaults to English ('en').
        translate (bool, optional): Indicates whether translation tasks are required (default is False).
        save_output_recording (bool, optional): Indicates whether to save recording from microphone.
        output_recording_filename (str, optional): File to save the output recording.
        output_transcription_path (str, optional): File to save the output transcription.

    Attributes:
        client (Client): An instance of the underlying Client class responsible for handling the WebSocket connection.

    Example:
        To create a TranscriptionClient and start transcription on microphone audio:
        ```python
        transcription_client = TranscriptionClient(host="localhost", port=9090)
        transcription_client()
        ```
    """

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
        server_command=None,
    ):
        # 如果配置了 server_command，先启动 server
        self._server_command = server_command
        self._server_process = None
        if server_command:
            self.start_server()

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
        if save_output_recording and not output_recording_filename.endswith(".wav"):
            raise ValueError(
                f"Please provide a valid `output_recording_filename`: {output_recording_filename}"
            )
        if not output_transcription_path.endswith(".srt"):
            raise ValueError(
                f"Please provide a valid `output_transcription_path`: {output_transcription_path}. The file extension should be `.srt`."
            )
        TranscriptionTeeClient.__init__(
            self,
            [self.client],
            save_output_recording=save_output_recording,
            output_recording_filename=output_recording_filename,
        )
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
