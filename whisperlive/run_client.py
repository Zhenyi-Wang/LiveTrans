import os

from whisper_live.client import TranscriptionClient

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
client(other_url="http://mini:8080/live/livestream.flv")
