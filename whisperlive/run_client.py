import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from whisper_live.client import TranscriptionClient

# /status 检查端口: mini 侧值守监控拉取转录运行状态(流三态/WS/最近字幕);
# 0 = 关闭。转录进程挂掉端口即消失, 监控据此发现"转录服务不可达"
STATUS_PORT = int(os.environ.get("LIVETRANS_STATUS_PORT", "9091"))

client = TranscriptionClient(
    "localhost",
    9090,
    model="/home/zhenyi/models/whispers/whisper-large-v2-finetune-no-timestamps-ct2-new",
    lang="zh",
    use_vad=True,
    # 分发目标: 生产mini / 本地dev测试改为 http://localhost:8081
    # (注: DISPATCH_API环境变量方案存在未定位的失灵问题,暂用字面量切换)
    dispatch_api="http://mini:8081/backend/api/listen",
    server_command=["python", "whisperlive/run_server.py", "--port", "9090", "--backend", "faster_whisper"],
)


def start_status_server(tee):
    if not STATUS_PORT:
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/status":
                self.send_error(404)
                return
            try:
                body = json.dumps(tee.status_snapshot()).encode()
            except Exception as e:
                body = json.dumps({"ok": False, "error": repr(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # 监控高频轮询, 不刷日志

    server = ThreadingHTTPServer(("0.0.0.0", STATUS_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[{client.__class__.__name__}] [STATUS] 检查端口已启动: http://0.0.0.0:{STATUS_PORT}/status")


start_status_server(client)
client(other_url="http://mini:8080/live/livestream.flv")
