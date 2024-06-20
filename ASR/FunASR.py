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
import os
import sys
sys.path.append('./')
from src.cost_time import calculate_time    

class FunASR:
    def __init__(self) -> None:
        # 定义模型的自定义路径
        model_path = "FunASR/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        vad_model_path = "FunASR/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        punc_model_path = "FunASR/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

        # 检查文件是否存在于 FunASR 目录下
        model_exists = os.path.exists(model_path)
        vad_model_exists = os.path.exists(vad_model_path)
        punc_model_exists = os.path.exists(punc_model_path)
        # Modelscope AutoDownload
        self.model = AutoModel(
            model=model_path if model_exists else "paraformer-zh",
            vad_model=vad_model_path if vad_model_exists else "fsmn-vad",
            punc_model=punc_model_path if punc_model_exists else "ct-punc-c",
        )
        # 自定义路径
        # self.model = AutoModel(model="FunASR/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", # model_revision="v2.0.4",
        #         vad_model="FunASR/speech_fsmn_vad_zh-cn-16k-common-pytorch", # vad_model_revision="v2.0.4",
        #         punc_model="FunASR/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", # punc_model_revision="v2.0.4",
        #         # spk_model="cam++", spk_model_revision="v2.0.2",
        #         )
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