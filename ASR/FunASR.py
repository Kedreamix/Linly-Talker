'''
Reference: https://github.com/alibaba-damo-academy/FunASR
pip install funasr
pip install modelscope
pip install -U rotary_embedding_torch
'''
try:
    from funasr import AutoModel
except:
    print("如果想使用FunASR，请先安装funasr，若使用Whisper，请忽略此条信息")
import sys
sys.path.append('..')
from src.cost_time import calculate_time    

class FunASR:
    def __init__(self) -> None:
        self.model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                # spk_model="cam++", spk_model_revision="v2.0.2",
                )
    @calculate_time
    def transcribe(self, audio_file):
        res = self.model.generate(input=audio_file, 
            batch_size_s=300)
        print(res)
        return res[0]['text']
    
        
if __name__ == "__main__":
    import os
    # 创建ASR对象并进行语音识别
    audio_file = "output.wav"  # 音频文件路径
    if not os.path.exists(audio_file):
        os.system('edge-tts --text "hello" --write-media output.wav')
    asr = FunASR()
    print(asr.transcribe(audio_file))