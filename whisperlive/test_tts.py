import torch
from TTS.api import TTS
from pydub import AudioSegment
import numpy as np

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models().list_models())
# exit(0)
available_models = [
    "tts_models/multilingual/multi-dataset/xtts_v2",
    "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "tts_models/multilingual/multi-dataset/your_tts",
    "tts_models/multilingual/multi-dataset/bark",
    "tts_models/bg/cv/vits",
    "tts_models/cs/cv/vits",
    "tts_models/da/cv/vits",
    "tts_models/et/cv/vits",
    "tts_models/ga/cv/vits",
    "tts_models/en/ek1/tacotron2",
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/tacotron2-DDC_ph",
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/ljspeech/speedy-speech",
    "tts_models/en/ljspeech/tacotron2-DCA",
    "tts_models/en/ljspeech/vits",
    "tts_models/en/ljspeech/vits--neon",
    "tts_models/en/ljspeech/fast_pitch",
    "tts_models/en/ljspeech/overflow",
    "tts_models/en/ljspeech/neural_hmm",
    "tts_models/en/vctk/vits",
    "tts_models/en/vctk/fast_pitch",
    "tts_models/en/sam/tacotron-DDC",
    "tts_models/en/blizzard2013/capacitron-t2-c50",
    "tts_models/en/blizzard2013/capacitron-t2-c150_v2",
    "tts_models/en/multi-dataset/tortoise-v2",
    "tts_models/en/jenny/jenny",
    "tts_models/es/mai/tacotron2-DDC",
    "tts_models/es/css10/vits",
    "tts_models/fr/mai/tacotron2-DDC",
    "tts_models/fr/css10/vits",
    "tts_models/uk/mai/glow-tts",
    "tts_models/uk/mai/vits",
    "tts_models/zh-CN/baker/tacotron2-DDC-GST",
    "tts_models/nl/mai/tacotron2-DDC",
    "tts_models/nl/css10/vits",
    "tts_models/de/thorsten/tacotron2-DCA",
    "tts_models/de/thorsten/vits",
    "tts_models/de/thorsten/tacotron2-DDC",
    "tts_models/de/css10/vits-neon",
    "tts_models/ja/kokoro/tacotron2-DDC",
    "tts_models/tr/common-voice/glow-tts",
    "tts_models/it/mai_female/glow-tts",
    "tts_models/it/mai_female/vits",
    "tts_models/it/mai_male/glow-tts",
    "tts_models/it/mai_male/vits",
    "tts_models/ewe/openbible/vits",
    "tts_models/hau/openbible/vits",
    "tts_models/lin/openbible/vits",
    "tts_models/tw_akuapem/openbible/vits",
    "tts_models/tw_asante/openbible/vits",
    "tts_models/yor/openbible/vits",
    "tts_models/hu/css10/vits",
    "tts_models/el/cv/vits",
    "tts_models/fi/css10/vits",
    "tts_models/hr/cv/vits",
    "tts_models/lt/cv/vits",
    "tts_models/lv/cv/vits",
    "tts_models/mt/cv/vits",
    "tts_models/pl/mai_female/vits",
    "tts_models/pt/cv/vits",
    "tts_models/ro/cv/vits",
    "tts_models/sk/cv/vits",
    "tts_models/sl/cv/vits",
    "tts_models/sv/cv/vits",
    "tts_models/ca/custom/vits",
    "tts_models/fa/custom/glow-tts",
    "tts_models/bn/custom/vits-male",
    "tts_models/bn/custom/vits-female",
    "tts_models/be/common-voice/glow-tts",
    "vocoder_models/universal/libri-tts/wavegrad",
    "vocoder_models/universal/libri-tts/fullband-melgan",
    "vocoder_models/en/ek1/wavegrad",
    "vocoder_models/en/ljspeech/multiband-melgan",
    "vocoder_models/en/ljspeech/hifigan_v2",
    "vocoder_models/en/ljspeech/univnet",
    "vocoder_models/en/blizzard2013/hifigan_v2",
    "vocoder_models/en/vctk/hifigan_v2",
    "vocoder_models/en/sam/hifigan_v2",
    "vocoder_models/nl/mai/parallel-wavegan",
    "vocoder_models/de/thorsten/wavegrad",
    "vocoder_models/de/thorsten/fullband-melgan",
    "vocoder_models/de/thorsten/hifigan_v1",
    "vocoder_models/ja/kokoro/hifigan_v1",
    "vocoder_models/uk/mai/multiband-melgan",
    "vocoder_models/tr/common-voice/hifigan",
    "vocoder_models/be/common-voice/hifigan",
    "voice_conversion_models/multilingual/vctk/freevc24",
]
test_models = [
    # "tts_models/multilingual/multi-dataset/xtts_v2",
    # "tts_models/multilingual/multi-dataset/xtts_v1.1",
    # "tts_models/multilingual/multi-dataset/your_tts",
    # "tts_models/multilingual/multi-dataset/bark",
    # "tts_models/en/ek1/tacotron2",
    # "tts_models/en/ljspeech/tacotron2-DDC",
    # "tts_models/en/ljspeech/tacotron2-DDC_ph",
    # "tts_models/en/ljspeech/glow-tts",
    # "tts_models/en/ljspeech/speedy-speech",
    "tts_models/en/ljspeech/tacotron2-DCA",
    # "tts_models/en/ljspeech/vits",
    # "tts_models/en/ljspeech/vits--neon",
    # "tts_models/en/ljspeech/fast_pitch",
    # "tts_models/en/ljspeech/overflow",
    # "tts_models/en/ljspeech/neural_hmm",
    # "tts_models/en/vctk/vits",
    # "tts_models/en/vctk/fast_pitch",
    # "tts_models/en/sam/tacotron-DDC",
    # "tts_models/en/blizzard2013/capacitron-t2-c50",
    # "tts_models/en/blizzard2013/capacitron-t2-c150_v2",
    # "tts_models/en/multi-dataset/tortoise-v2",
    # "tts_models/en/jenny/jenny",
]

for model in test_models:
    print(f"------Testing {model}------")
    try:
        # Init TT
        # /home/zhenyi/.local/share/tts/
        tts = TTS(model).to(device)
        # use model name as file name, replace "/" with "_"
        file_name = model.replace("/", "_")

        # # Run TTS
        # # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
        # # Text to speech list of amplitude values as output
        # wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
        # Text to speech to a file
        # tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
        #         tts.tts_to_file(
        #             # text="Trust in the Lord with all your heart, and lean not on your own understanding. In all your ways acknowledge Him, and He shall direct your paths.",
        #             text="""Dear brothers and sisters in Christ,

        # Today, let us reflect on the profound words of Jesus: "Blessed are the peacemakers, for they will be called children of God" (Matthew 5:9). In a world often torn by conflict and division, the role of the peacemaker is not only noble but essential. Peacemakers are not merely those who avoid strife; they are active agents of reconciliation, seeking harmony where there is discord. They embody the spirit of love and forgiveness that Jesus taught us.

        # As we navigate our daily lives, let us strive to be peacemakers in our homes, workplaces, and communities. It begins with a heart transformed by the grace of God, a heart that seeks to understand rather than to be understood, to love rather than to judge. When we encounter disagreements or misunderstandings, let us respond with patience and compassion, remembering that we are all children of the same heavenly Father.

        # Let us pray for the strength to be instruments of peace, to sow seeds of harmony, and to build bridges of understanding. In doing so, we not only honor God but also contribute to the well-being of our world. May we all be inspired by the example of Jesus, the ultimate peacemaker, and may our lives reflect His love and peace. Amen.

        # """,
        #             file_path=f"files/{file_name}.wav",
        #         )

        text = """Dear brothers and sisters in Christ,

        Today, let us reflect on the profound words of Jesus: "Blessed are the peacemakers, for they will be called children of God" (Matthew 5:9). In a world often torn by conflict and division, the role of the peacemaker is not only noble but essential. Peacemakers are not merely those who avoid strife; they are active agents of reconciliation, seeking harmony where there is discord. They embody the spirit of love and forgiveness that Jesus taught us.

        As we navigate our daily lives, let us strive to be peacemakers in our homes, workplaces, and communities. It begins with a heart transformed by the grace of God, a heart that seeks to understand rather than to be understood, to love rather than to judge. When we encounter disagreements or misunderstandings, let us respond with patience and compassion, remembering that we are all children of the same heavenly Father.

        Let us pray for the strength to be instruments of peace, to sow seeds of harmony, and to build bridges of understanding. In doing so, we not only honor God but also contribute to the well-being of our world. May we all be inspired by the example of Jesus, the ultimate peacemaker, and may our lives reflect His love and peace. Amen.

        """

        wav = tts.tts(text)
        import json
        import numpy as np

        wav_np = np.array(wav)
        print(len(wav_np))
        print(len(wav_np.tobytes()))
        print(len(json.dumps({"audio": wav_np})))
        break

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

        print(len(audio_segment))

        break
        audio_segment.export(f"files/default.mp3", format="mp3")

        # 动态调整音量到合适的倍数
        audio_segment = audio_segment.normalize()

        audio_segment.export(f"files/nomalized.mp3", format="mp3")

    except Exception as e:
        print(f"Error with {model}: {e}")
