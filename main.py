import subprocess
import time
import signal
import GPUtil
import psutil

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
            process = psutil.Process(pid)
            process.terminate()
            print(f"Terminated process with PID {pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            print(f"Failed to terminate process with PID {pid}")
    pids.clear()

# 信号处理函数
def signal_handler(signum, frame):
    print("Received SIGINT, terminating subprocess...")
    kill_processes()
    print("Exiting...")
    exit(1)

# 注册信号处理函数
signal.signal(signal.SIGINT, signal_handler)


def restart_scripts():
    kill_processes()
    run_script("run_server.py")
    time.sleep(5)
    run_script("run_client.py")

def get_gpu_utilization():
    import re

    # 执行nvidia-smi命令
    smi_output = subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8")

    # 找到包含"184"的行
    lines = smi_output.split("\n")
    line_with_184 = next((line for line in lines if "184" in line), None)

    # # 在找到的行中匹配第一个百分数
    # match = re.search(r"\d+%", line_with_184)
    # percentage_str = match.group(0)[:-1]
    # return int(percentage_str)

    # 在找到的行中匹配第一个"P"后面跟数字的字符串
    match = re.search(r"P\d+", line_with_184)
    if match:
        return match.group(0)
    else:
        return "P2"

p8_count = 0 # 记录P8的数量
p8_thres = 2 # P8重启的阈值

def monitor_gpu():
    restart_scripts()
    time.sleep(120)
    while True:
        gpu_status = get_gpu_utilization()
        if gpu_status == "P8":
            p8_count += 1
            if p8_count >= p8_thres:
                print("*" * 50)
                print("*", "GPU stopped, restarting scripts...")
                print("*" * 50)
                restart_scripts()
                time.sleep(120)  # Wait for scripts to start up
            else:
                print("*" * 30, f"GPU P8 detected, count {p8_count} / {p8_thres}")
        else:
            p8_count = 0
            print("*" * 30, f"GPU status is {gpu_status}")

        time.sleep(5)


if __name__ == "__main__":
    monitor_gpu()
    # restart_scripts()
