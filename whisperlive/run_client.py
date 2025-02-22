# python3 run_server.py --port 9090 --backend faster_whisper

from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
    "localhost",
    9090,
    # model="/home/zhenyi/ownprojects/livetrans/models/faster-whisper-large-v2",
    model="large-v2",
    # model="/home/zhenyi/ownprojects/livetrans/models/faster-whisper-large-v3-turbo-ct2",
    # model="/home/zhenyi/models/transCT2/Belle-whisper-large-v3-turbo-zh-ct2",
    # model="/home/zhenyi/models/transCT2/Belle-distilwhisper-large-v2-zh-ct2",
    # model="/home/zhenyi/models/transCT2/Belle-whisper-large-v3-zh-punct-ct2",
    # model="/home/zhenyi/models/transCT2/Belle-whisper-large-v2-zh-ct2",
    # model="large-v3",
    lang="zh",
    use_vad=True,
    # dispatch_api="http://localhost:8081/backend/api/listen",
    dispatch_api="http://mini:8081/backend/api/listen",
)
# client(other_url="https://realtime.hainingchurch.cn/live/livestream.m3u8")
client(other_url="https://realtime.hainingchurch.cn/live/livestream.flv")
