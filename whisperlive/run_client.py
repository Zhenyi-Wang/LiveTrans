import json
import os
import threading
import urllib.parse
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
        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/status":
                self.send_error(404)
                return
            try:
                self._send_json(200, tee.status_snapshot())
            except Exception as e:
                self._send_json(200, {"ok": False, "error": repr(e)})

        def do_POST(self):
            # 暂停/恢复控制入口(参数走 query string, .bat/curl 最简); body 读取丢弃防 keep-alive 挂起。
            # 不加鉴权: 暴露面与 /status 相同(LAN+WG 可信网络)
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length:
                self.rfile.read(length)
            try:
                if parsed.path == "/pause":
                    params = urllib.parse.parse_qs(parsed.query)
                    hours = float(params.get("hours", ["2"])[0])  # 无参默认 2h, 支持小数
                    resume_at = tee.pause(hours)
                    self._send_json(200, {
                        "ok": True,
                        "paused": True,
                        "resume_at": int(resume_at * 1000),
                        "resume_at_iso": tee._fmt_resume(),
                    })
                elif parsed.path == "/resume":
                    tee.resume()  # 幂等: 未暂停时调用无害
                    self._send_json(200, {"ok": True, "paused": False})
                else:
                    self._send_json(404, {"ok": False, "error": "not found"})
            except ValueError as e:  # hours 非数字/超范围
                self._send_json(400, {"ok": False, "error": str(e)})
            except Exception as e:
                self._send_json(500, {"ok": False, "error": repr(e)})

        def log_message(self, *args):
            pass  # 监控高频轮询, 不刷日志

    server = ThreadingHTTPServer(("0.0.0.0", STATUS_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[{client.__class__.__name__}] [STATUS] 检查端口已启动: http://0.0.0.0:{STATUS_PORT}/status")


start_status_server(client)
client(other_url="http://mini:8080/live/livestream.flv")
