import os

from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    "localhost",
    9090,
    model="/home/zhenyi/models/whispers/whisper-large-v2-finetune-no-timestamps-ct2-new",
    lang="zh",
    use_vad=True,
    dispatch_api="http://mini:8081/backend/api/listen",
    server_command=["python", "whisperlive/run_server.py", "--port", "9090", "--backend", "faster_whisper"],
)
client(other_url="http://mini:8080/live/livestream.flv")
