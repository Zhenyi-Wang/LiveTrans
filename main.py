import subprocess
import threading
import time
import signal
import sys
from datetime import datetime

exit_requested = False

_ALSA_NOISE = ("ALSA lib", "Cannot get card", "Cannot open device",
               "Unknown PCM", "Invalid card", "Invalid field",
               "pcm_oss", "pcm_usb", "snd_pcm", "snd_func",
               "snd_config", "_snd_pcm", "Evaluate error",
               "BuildDeviceList", "Assertion")


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def signal_handler(signum, frame):
    global exit_requested
    exit_requested = True
    print(f"[{ts()}] [MAIN] Received signal {signum}, waiting for client to exit...")


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
    print(f"[{ts()}] [MAIN] Starting client process")
    process = subprocess.Popen(
        [sys.executable, "whisperlive/run_client.py"],
        stderr=subprocess.PIPE,
    )
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
            break

        print(f"[{ts()}] [MAIN] Client crashed (exit_code={returncode}), restarting in {restart_delay}s (attempt #{attempt})")
        time.sleep(restart_delay)
        restart_delay = min(restart_delay * 2, max_restart_delay)
