import subprocess
import time
import signal

# 全局变量，用于存储PID
pids = []


def run_script(script_name):
    # 运行脚本并获取其 PID
    process = subprocess.Popen(["python", script_name])
    pid = process.pid
    print(f"Started {script_name} with PID {pid}")
    pids.append(pid)


def kill_processes():
    for pid in pids:
        try:
            import psutil
            process = psutil.Process(pid)
            process.terminate()
            print(f"Terminated process with PID {pid}")
        except Exception:
            pass
    pids.clear()


def signal_handler(signum, frame):
    print("Received SIGINT, terminating subprocess...")
    kill_processes()
    print("Exiting...")
    exit(1)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    # client 会自动管理 server 的生命周期
    run_script("whisperlive/run_client.py")
