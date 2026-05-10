import subprocess
import time
import signal
import sys
from datetime import datetime

exit_requested = False


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def signal_handler(signum, frame):
    global exit_requested
    exit_requested = True
    print(f"[{ts()}] [MAIN] Received signal {signum}, waiting for client to exit...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_client():
    """运行 client 进程，返回 (process, returncode)。"""
    print(f"[{ts()}] [MAIN] Starting client process")
    process = subprocess.Popen(
        [sys.executable, "whisperlive/run_client.py"],
    )
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
