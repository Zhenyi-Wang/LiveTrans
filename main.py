import os
import subprocess
import threading
import time
import signal
import sys
from datetime import datetime

SERVER_PORT = 9090  # run_server.py 监听端口，client 卡死被强杀时需兜底清理

exit_requested = False
client_proc = None
sig_count = 0

_ALSA_NOISE = ("ALSA lib", "Cannot get card", "Cannot open device",
               "Unknown PCM", "Invalid card", "Invalid field",
               "pcm_oss", "pcm_usb", "snd_pcm", "snd_func",
               "snd_config", "_snd_pcm", "Evaluate error",
               "BuildDeviceList", "Assertion")


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _kill_port_users(port):
    """杀掉占用端口的残留进程（与 client._kill_port_users 同逻辑）。

    client 被 SIGTERM/强杀时不会走自己的 finally 清理，server 会变孤儿
    继续占端口和显存，退出前在这里兜底。"""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for pid in result.stdout.split():
            print(f"[{ts()}] [MAIN] Killing stale process on port {port} (pid={pid})")
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
    except Exception:
        pass


def signal_handler(signum, frame):
    global exit_requested, sig_count
    sig_count += 1
    exit_requested = True
    proc = client_proc
    if proc is None or proc.poll() is not None:
        return
    if sig_count == 1:
        # 第一次：Ctrl+C 时 client 同样收到 SIGINT，其 finally 会写 SRT 并优雅停
        # server，先给它时间自己退；卡死在不可中断调用里时（历次"Ctrl+C 停不下"
        # 的根因）再由监督进程接管
        print(f"[{ts()}] [MAIN] Received signal {signum}, waiting for client to exit...")

        def _terminate():
            if proc.poll() is None:
                print(f"[{ts()}] [MAIN] Client still alive after 5s, terminating")
                proc.terminate()

        def _force_kill():
            if proc.poll() is None:
                print(f"[{ts()}] [MAIN] Client not responding to SIGTERM, force killing")
                proc.kill()

        threading.Timer(5.0, _terminate).start()
        threading.Timer(12.0, _force_kill).start()
    else:
        # 再次 Ctrl+C：不再等优雅退出，立即终止
        print(f"[{ts()}] [MAIN] Received signal {signum} again, stopping now...")
        proc.terminate()

        def _force_kill_now():
            if proc.poll() is None:
                proc.kill()

        threading.Timer(2.0, _force_kill_now).start()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _stderr_reader(proc):
    """后台线程：过滤 client 子进程的 ALSA 噪音。"""
    try:
        for line in proc.stderr:
            text = line.decode("utf-8", errors="replace").rstrip()
            if any(text.startswith(p) or text.startswith(f"python: {p}") for p in _ALSA_NOISE):
                continue
            sys.stderr.write(text + "\n")
            sys.stderr.flush()
    except Exception:
        pass


def run_client():
    """运行 client 进程，返回 (process, returncode)。"""
    global client_proc
    print(f"[{ts()}] [MAIN] Starting client process")
    process = subprocess.Popen(
        [sys.executable, "whisperlive/run_client.py"],
        stderr=subprocess.PIPE,
    )
    client_proc = process
    threading.Thread(target=_stderr_reader, args=(process,), daemon=True).start()
    return process


if __name__ == "__main__":
    max_restart_delay = 60  # 最大重启等待间隔
    restart_delay = 5        # 初始重启等待间隔
    attempt = 0

    print(f"[{ts()}] [MAIN] livetrans supervisor started")

    while not exit_requested:
        attempt += 1
        process = run_client()
        returncode = process.wait()

        if exit_requested:
            print(f"[{ts()}] [MAIN] Clean exit")
            time.sleep(2)  # 给 client 自己的 finally 一点时间停 server
            _kill_port_users(SERVER_PORT)
            break

        print(f"[{ts()}] [MAIN] Client crashed (exit_code={returncode}), restarting in {restart_delay}s (attempt #{attempt})")
        time.sleep(restart_delay)
        restart_delay = min(restart_delay * 2, max_restart_delay)
