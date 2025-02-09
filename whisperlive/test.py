# import requests
# import asyncio


# class LLM:
#     def __init__(self):
#         self.key = "sk-a5c8b88f13c54074acaf785d1c43b59c"
#         self.url = "https://api.deepseek.com/chat/completions"

#     async def query(self, prompt, content):
#         response = requests.post(self.url, headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.key}"
#         },
#         json={
#             "model":"deepseek-chat",
#             "messages":[
#                 {"role":"system", "content":prompt},
#                 {"role":"user", "content":content}
#             ],
#             "stream":False
#         })

#         return response.json()['choices'][0]['message']['content']

# async def main():
#     llm = LLM()
#     prompt = "you are a helpful assistant"
#     content = "你好"
#     response = await llm.query(prompt, content)
#     print(response)

# # 运行异步主函数
# asyncio.run(main())


import time

import librosa
import numpy as np

from whisper_live.transcriber import WhisperModel

def render_offset(offset): 
    return f"{(offset//60)%60:02}:{offset%60:02}"

offset = 17*60 + 34 + 5 + 5 + 4 + 4 + 2 + 9 + 8 + 7 + 28 + 5.6 + 2.8 + 4.4 + 5 + 35.44 +23 + 6 + 5.6 + 4.5 + 6
# offset = 4* 60 + 24.62 + 11.72 + 12 + 23 + 12 + 24.76 + 5 + 7.6 + 7
duration = 120
sr = 16000
y, sr = librosa.load("files/6-30.mp3", sr=sr)
input_sample = np.array(y, dtype=np.float32)[int(offset * sr) : int((offset + duration) * sr)]
print("audio loaded", input_sample.shape[0] / sr, "seconds long, offset:", render_offset(offset))

model = WhisperModel(
    # "large-v3",
    "large-v2",
    # "medium",
    device="cuda",
    compute_type="int8",  #  if device == "cpu" else "float16",
    local_files_only=True,
)

print("model loaded")
start_time = time.time()
result, info = model.transcribe(
    input_sample,
    initial_prompt="基督。",
    language="zh",
    # task=self.task,
    vad_filter=True,
    # vad_parameters=self.vad_parameters if self.use_vad else None,
    # chunk_length=10,
    # patience=1,
)
if result:
    for s in result:
        # print(s.start, s.end, s.text)
        print(s)
        print('---')
# print(info)
print(
    f"time ratio: {duration} / {round(time.time() - start_time,1)} = {round(duration / (time.time() - start_time),1)}"
)
