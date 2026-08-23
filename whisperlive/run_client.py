import os

from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    "localhost",
    9090,
    model="/home/zhenyi/models/whispers/whisper-large-v2-finetune-no-timestamps-ct2-new",
    lang="zh",
    use_vad=True,
    # 分发目标: 默认生产(mini),本地测试 DISPATCH_API=http://localhost:8081 ./start.sh
    dispatch_api=os.environ.get("DISPATCH_API", "http://mini:8081/backend/api/listen"),
    server_command=["python", "whisperlive/run_server.py", "--port", "9090", "--backend", "faster_whisper"],
)
client(other_url="http://mini:8080/live/livestream.flv")
