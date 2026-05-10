import subprocess
import time
import signal
import sys

exit_requested = False


def signal_handler(signum, frame):
    global exit_requested
    exit_requested = True
    print("Received SIGINT, waiting for client to exit...")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_client():
    """运行 client 进程，返回退出码。"""
    process = subprocess.Popen(
        [sys.executable, "whisperlive/run_client.py"],
    )
    return process


if __name__ == "__main__":
    max_restart_delay = 60  # 最大重启等待间隔
    restart_delay = 5        # 初始重启等待间隔

    while not exit_requested:
        process = run_client()
        returncode = process.wait()

        if exit_requested:
            break

        print(f"[WARN] Client exited with code {returncode}, restarting in {restart_delay}s...")
        time.sleep(restart_delay)
        restart_delay = min(restart_delay * 2, max_restart_delay)
