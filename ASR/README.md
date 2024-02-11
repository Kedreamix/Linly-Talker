## ASR 同数字人沟通的桥梁

### Whisper OpenAI

Whisper 是一个自动语音识别 (ASR) 系统，它使用从网络上收集的 680,000 小时多语言和多任务监督数据进行训练。使用如此庞大且多样化的数据集可以提高对口音、背景噪音和技术语言的鲁棒性。此外，它还支持多种语言的转录，以及将这些语言翻译成英语。

使用方法很简单，我们只要安装以下库，后续模型会自动下载

```bash
pip install -U openai-whisper
```

借鉴OpenAI的Whisper实现了ASR的语音识别，具体使用方法参考 [https://github.com/openai/whisper](https://github.com/openai/whisper)

```python
'''
https://github.com/openai/whisper
pip install -U openai-whisper
'''
import whisper

class WhisperASR:
    def __init__(self, model_path):
        self.LANGUAGES = {
            "en": "english",
            "zh": "chinese",
        }
        self.model = whisper.load_model(model_path)
        
    def transcribe(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"]
```



### FunASR Alibaba

阿里的`FunASR`的语音识别效果也是相当不错，而且时间也是比whisper更快的，更能达到实时的效果，所以也将FunASR添加进去了，在ASR文件夹下的FunASR文件里可以进行体验，参考 [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR)

需要注意的是，在第一次运行的时候，需要安装以下库。

```bash
pip install funasr
pip install modelscope
pip install -U rotary_embedding_torch
```

```python
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

class FunASR:
    def __init__(self) -> None:
        self.model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                # spk_model="cam++", spk_model_revision="v2.0.2",
                )

    def transcribe(self, audio_file):
        res = self.model.generate(input=audio_file, 
            batch_size_s=300)
        print(res)
        return res[0]['text']
```



