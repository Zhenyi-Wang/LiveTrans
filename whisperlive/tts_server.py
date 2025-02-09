from fastapi import FastAPI

from fastapi import FastAPI
from TTS.api import TTS
import uvicorn
import uuid
from pydub import AudioSegment
import numpy as np


# device = "cuda" if torch.cuda.is_available() else "cpu"

tts_model = TTS(
    "tts_models/en/ljspeech/tacotron2-DCA",
    vocoder_path="vocoder_models/en/ek1/wavegrad",
).to("cuda")
# load models
# synthesizer = Synthesizer(
#     tts_checkpoint="/home/zhenyi/.local/share/tts/tts_models--en--ljspeech--tacotron2-DCA/model_file.pth",
#     tts_config_path="/home/zhenyi/.local/share/tts/tts_models--en--ljspeech--tacotron2-DCA/config.json",
#     use_cuda=True,
# )

# app = FastAPI(lifespan=lifespan)
app = FastAPI()


@app.get(
    "/tts",
    responses={
        200: {
            "description": "",
        },
    },
)
async def tts(text):
    # wavs = synthesizer.tts(text)
    name = str(uuid.uuid4())
    # synthesizer.save_wav(wavs, f"files/{name}.wav")
    wav = tts_model.tts(text)

    # print(wav[:100])
    # 将浮点数列表转换为 numpy 数组
    audio_data = np.array(wav)

    # 将浮点数缩放到 16 位整数范围 (-32768 到 32767)
    audio_data = (audio_data * 32767).astype(np.int16)

    # print(audio_data[:100])

    audio_segment = AudioSegment(
        audio_data.tobytes(),  # 将numpy数组转换为字节
        frame_rate=22050,  # 设置采样率
        sample_width=2,  # 设置样本宽度（2字节表示16位）
        channels=1,  # 设置通道数（单声道）
    )

    # # 增加音量 1.5 倍
    # gain_value = 3.5218  # 增加 1.5 倍的音量对应的增益值（以 dB 为单位）
    # audio_segment = audio_segment.apply_gain(gain_value)

    # 动态调整音量到合适的倍数
    audio_segment = audio_segment.normalize()

    audio_segment.export(f"files/{name}.mp3", format="mp3")

    # tts_model.tts_to_file(
    #     text=text,
    #     file_path=f"files/{name}.wav",
    # )

    return {"id": name}


if __name__ == "__main__":
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8082, reload=True)
