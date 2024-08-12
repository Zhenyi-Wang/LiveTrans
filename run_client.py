# python3 run_server.py --port 9090 --backend faster_whisper

from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    "localhost",
    9090,
    # model="./models/faster-whisper-large-v2",
    model="large-v2",
    # model="large-v3",
    lang="zh",
    use_vad=True,
    # dispatch_api="http://localhost:8081/listen",
    dispatch_api="http://mini:8081/listen",
)
# client(other_url="https://stream.hainingchurch.cn:38080/live/livestream.m3u8")
client(other_url="https://stream.hainingchurch.cn:38080/live/livestream.flv")
