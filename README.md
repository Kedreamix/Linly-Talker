# 数字人对话系统 - Linly-Talker —— “数字人交互，与虚拟的自己互动”

<div align="center">
<h1>Linly-Talker WebUI</h1>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/Kedreamix/Linly-Talker)

<img src="https://github.com/CVI-SZU/Linly/raw/main/assets/linly_logo.png" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/Kedreamix/Linly-Talker/blob/main/colab_webui.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/Kedreamix/Linly-Talker/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models%20Repo-yellow.svg?style=for-the-badge)](https://huggingface.co/Kedreamix/Linly-Talker)

[**English**](./README_en.md) | [**中文简体**](./README.md)

</div>

**2023.12 更新** 📆

**用户可以上传任意图片进行对话**

**2024.01 更新** 📆

- **令人兴奋的消息！我现在已经将强大的GeminiPro和Qwen大模型融入到我们的对话场景中。用户现在可以在对话中上传任何图片，为我们的互动增添了全新的层面。**
- **更新了FastAPI的部署调用方法。** 
- **更新了微软TTS的高级设置选项，增加声音种类的多样性，以及加入视频字幕加强可视化。**
  - **更新了GPT多轮对话系统，使得对话有上下文联系，提高数字人的交互性和真实感。**


**2024.02 更新** 📆

- **更新了Gradio的版本为最新版本4.16.0，使得界面拥有更多的功能，比如可以摄像头拍摄图片构建数字人等。**
- **更新了ASR和THG，其中ASR加入了阿里的FunASR，具体更快的速度；THG部分加入了Wav2Lip模型，ER-NeRF在准备中(Comming Soon)。**
- **加入了语音克隆方法GPT-SoVITS模型，能够通过微调一分钟对应人的语料进行克隆，效果还是相当不错的，值得推荐。**
- **集成一个WebUI界面，能够更好的运行Linly-Talker。**

---



## 介绍

Linly-Talker是一个将大型语言模型与视觉模型相结合的智能AI系统,创建了一种全新的人机交互方式。它集成了各种技术,例如Whisper、Linly、微软语音服务和SadTalker会说话的生成系统。该系统部署在Gradio上,允许用户通过提供图像与AI助手进行交谈。用户可以根据自己的喜好进行自由的对话或内容生成。

![The system architecture of multimodal human–computer interaction.](docs/HOI.png)

> 查看我们的介绍视频 [demo video](https://www.bilibili.com/video/BV1rN4y1a76x/)

## TO DO LIST

- [x] 基本完成对话系统流程，能够`语音对话`
- [x] 加入了LLM大模型，包括`Linly`，`Qwen`和`GeminiPro`的使用
- [x] 可上传`任意数字人照片`进行对话
- [x] Linly加入`FastAPI`调用方式
- [x] 利用微软`TTS`加入高级选项，可设置对应人声以及音调等参数，增加声音的多样性
- [x] 视频生成加入`字幕`，能够更好的进行可视化
- [x] GPT`多轮对话`系统（提高数字人的交互性和真实感，增强数字人的智能）
- [x] 优化Gradio界面，加入更多模型，如Wav2Lip，FunASR等
- [x] `语音克隆`技术，加入GPT-SoVITS，只需要一分钟的语音简单微调即可（语音克隆合成自己声音，提高数字人分身的真实感和互动体验）
- [ ] 加入`Langchain`的框架，建立本地知识库
- [ ] `实时`语音识别（人与数字人之间就可以通过语音进行对话交流)

🔆 该项目 Linly-Talker 正在进行中 - 欢迎提出PR请求！如果您有任何关于新的模型方法、研究、技术或发现运行错误的建议，请随时编辑并提交 PR。您也可以打开一个问题或通过电子邮件直接联系我。📩⭐ 如果您发现这个Github Project有用，请给它点个星！🤩

## 示例

|                        文字/语音对话                         |                          数字人回答                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                 应对压力最有效的方法是什么？                 | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/f1deb189-b682-4175-9dea-7eeb0fb392ca"></video> |
|                      如何进行时间管理？                      | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/968b5c43-4dce-484b-b6c6-0fd4d621ac03"></video> |
|  撰写一篇交响乐音乐会评论，讨论乐团的表演和观众的整体体验。  | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/f052820f-6511-4cf0-a383-daf8402630db"></video> |
| 翻译成中文：Luck is a dividend of sweat. The more you sweat, the luckier you get. | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/118eec13-a9f7-4c38-b4ad-044d36ba9776"></video> |

## 创建环境

```bash
conda create -n linly python=3.10  
conda activate linly

# pytorch安装方式1：conda安装（推荐）
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# pytorch安装方式2：pip 安装
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install -q ffmpeg # ffmpeg==4.2.2

pip install -r requirements_app.txt
```

若使用语音克隆等模型，需要更高版本的Pytorch，但是功能也会更加丰富，不过需要的驱动版本可能要到cuda11.8，可选择

```bash
conda create -n linly python=3.10  
conda activate linly

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

conda install -q ffmpeg # ffmpeg==4.2.2

pip install -r requirements_app.txt

# 安装语音克隆对应的依赖
pip install -r VITS/requirements_gptsovits.txt
```

为了大家的部署使用方便，更新了一个`configs.py`文件，可以对其进行一些超参数修改即可

```bash
# 设备运行端口 (Device running port)
port = 7860
# api运行端口及IP (API running port and IP)
mode = 'api' # api 需要先运行Linly-api-fast.py，暂时仅仅适用于Linly
ip = '127.0.0.1' 
api_port = 7871

# L模型路径 (Linly model path)
mode = 'offline'
model_path = 'Qwen/Qwen-1_8B-Chat'

# ssl证书 (SSL certificate) 麦克风对话需要此参数
# 最好调整为绝对路径
ssl_certfile = "./https_cert/cert.pem"
ssl_keyfile = "./https_cert/key.pem"
```

## ASR - Speech Recognition

### Whisper

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



### FunASR

阿里的`FunASR`的语音识别效果也是相当不错，而且时间也是比whisper更快的，更能达到实时的效果，所以也将FunASR添加进去了，在ASR文件夹下的FunASR文件里可以进行体验，需要注意的是，在第一次运行的时候，需要安装以下库，参考 [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR)

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



## TTS - Edge TTS

使用微软语音服务,具体使用方法参考[https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)

我编写了一个 `EdgeTTS` 的类，能够更好的使用，并且增加了保存字幕文件的功能

```python
class EdgeTTS:
    def __init__(self, list_voices = False, proxy = None) -> None:
        voices = list_voices_fn(proxy=proxy)
        self.SUPPORTED_VOICE = [item['ShortName'] for item in voices]
        self.SUPPORTED_VOICE.sort(reverse=True)
        if list_voices:
            print(", ".join(self.SUPPORTED_VOICE))

    def preprocess(self, rate, volume, pitch):
        if rate >= 0:
            rate = f'+{rate}%'
        else:
            rate = f'{rate}%'
        if pitch >= 0:
            pitch = f'+{pitch}Hz'
        else:
            pitch = f'{pitch}Hz'
        volume = 100 - volume
        volume = f'-{volume}%'
        return rate, volume, pitch

    def predict(self,TEXT, VOICE, RATE, VOLUME, PITCH, OUTPUT_FILE='result.wav', OUTPUT_SUBS='result.vtt', words_in_cue = 8):
        async def amain() -> None:
            """Main function"""
            rate, volume, pitch = self.preprocess(rate = RATE, volume = VOLUME, pitch = PITCH)
            communicate = Communicate(TEXT, VOICE, rate = rate, volume = volume, pitch = pitch)
            subs: SubMaker = SubMaker()
            sub_file: Union[TextIOWrapper, TextIO] = (
                open(OUTPUT_SUBS, "w", encoding="utf-8")
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # audio_file.write(chunk["data"])
                    pass
                elif chunk["type"] == "WordBoundary":
                    # print((chunk["offset"], chunk["duration"]), chunk["text"])
                    subs.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])
            sub_file.write(subs.generate_subs(words_in_cue))
            await communicate.save(OUTPUT_FILE)
            
        
        # loop = asyncio.get_event_loop_policy().get_event_loop()
        # try:
        #     loop.run_until_complete(amain())
        # finally:
        #     loop.close()
        asyncio.run(amain())
        with open(OUTPUT_SUBS, 'r', encoding='utf-8') as file:
            vtt_lines = file.readlines()

        # 去掉每一行文字中的空格
        vtt_lines_without_spaces = [line.replace(" ", "") if "-->" not in line else line for line in vtt_lines]
        # print(vtt_lines_without_spaces)
        with open(OUTPUT_SUBS, 'w', encoding='utf-8') as output_file:
            output_file.writelines(vtt_lines_without_spaces)
        return OUTPUT_FILE, OUTPUT_SUBS
```

同时在`src`文件夹下，写了一个简易的`WebUI`

```bash
python TTS_app.py
```

![TTS](docs/TTS.png)

## Voice Clone

### GPT-SoVITS（推荐）

感谢大家的开源贡献，我借鉴了当前开源的语音克隆模型 `GPT-SoVITS`，我认为效果是相当不错的，项目地址可参考[https://github.com/RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

他有以下功能：

1. **零样本文本到语音（TTS）：** 输入 5 秒的声音样本，即刻体验文本到语音转换。
2. **少样本 TTS：** 仅需 1 分钟的训练数据即可微调模型，提升声音相似度和真实感。
3. **跨语言支持：** 支持与训练数据集不同语言的推理，目前支持英语、日语和中文。
4. **WebUI 工具：** 集成工具包括声音伴奏分离、自动训练集分割、中文自动语音识别(ASR)和文本标注，协助初学者创建训练数据集和 GPT/SoVITS 模型。

之前很多方法都是少样本，比如`OpenVoice`和`XTTS`，我之前也想着使用他们来进行实现语音克隆部分，但是很遗憾的是，并没有感觉有很好的效果，其实`XTTS`还是不错的，如果我们简单用麦克风🎤说几句话作为参考来进行克隆，我觉得效果还是可以的。

但是如果遇到比较高的要求，我觉得可能就需要更好的模型，并且成本也要打压下来，所以我就看到了这个`GPT-SoVITS`，我觉得这个模型是相当厉害的，少样本的TTS能做，也能做跨语言支持，这样我们很有可能就可以体验到奥巴马讲中文之类的，这样就可以完成视频翻译的一些任务了，所以我是很推崇这样的简单微调，效果又好的方法的。

为了尊重作者，我并没有把`GPT-SoVITS`的全套代码搬过来，我写了一个关于语音克隆的类，大家可以将训练好的模型参数中，就可以在本项目使用经过语音克隆后的TTS了，希望大家玩的开心，玩的愉快。

> 如果使用语音克隆模型，可能需要python为3.10，pytorch为2.1左右可能比较好，我的环境已经测试过了，简单来说，先安装GPT-SoVITS的环境，再直接pip intsall -r requirements_app.txt即可使用

```python
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# 安装对应的依赖
pip install -r VITS/requirements_gptsovits.txt

# 启动如下的WebUI界面
python VITS/app.py 
```

![](docs/GPT-SoVITS.png)

### XTTS

Coqui XTTS是一个领先的深度学习文本到语音任务（TTS语音生成模型）工具包，通过使用一段5秒钟以上的语音频剪辑就可以完成声音克隆*将语音克隆到不同的语言*。支持多种语言文本到语音转换，使其成为国际化应用的理想选择，这一特点特别适用于全球化的市场，其中需要生成多种语言的语音内容。所以在实验过程中，我也加入了这一部分，不过暂时使用的是默认的模型，并没有进行微调，个人认为是没有GPT-SoVITS经过微调后好的，但是其中的少样本五秒钟克隆语音还是值得称赞的。大家也可以在官方的在线体验，但是官方的可能会有生成语音限制，文字不能太长，但是还是足够我们体验了。

🐸TTS 是一个用于高级文本转语音生成的库。

🚀 超过 1100 种语言的预训练模型。

🛠️ 用于以任何语言训练新模型和微调现有模型的工具。

📚 用于数据集分析和管理的实用程序。

- 在线体验XTTS [https://huggingface.co/spaces/coqui/xtts](https://huggingface.co/spaces/coqui/xtts)
- 官方Github库 https://github.com/coqui-ai/TTS

XTTS的环境也需要PyTorch 2.1所以，如果下载了GPT-SoVITS，也不妨体验一下XTTS的效果。

```bash
python VITS/XTTS.py
```

![](docs/XTTS.png)



## THG - Avatar

### SadTalker

数字人生成可使用SadTalker（CVPR 2023）,详情介绍见 [https://sadtalker.github.io](https://sadtalker.github.io)

在使用前先下载SadTalker模型:

```bash
bash scripts/sadtalker_download_models.sh  
```

[Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)

> 如果百度网盘下载，记住是放在checkpoints文件夹下，百度网盘下载的默认命名为sadtalker，实际应该重命名为checkpoints

### Wav2Lip

数字人生成还可使用Wav2Lip（ACM 2020），详情介绍见 [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)

在使用前先下载Wav2Lip模型：

| Model                        | Description                                           | Link to the model                                            |
| ---------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Wav2Lip                      | Highly accurate lip-sync                              | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW) |
| Wav2Lip + GAN                | Slightly inferior lip-sync, but better visual quality | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |
| Expert Discriminator         | Weights of the expert discriminator                   | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP) |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup     | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo) |

```python
class Wav2Lip:
    def __init__(self, path = 'checkpoints/wav2lip.pth'):
        self.fps = 25
        self.resize_factor = 1
        self.mel_step_size = 16
        self.static = False
        self.img_size = 96
        self.face_det_batch_size = 2
        self.box = [-1, -1, -1, -1]
        self.pads = [0, 10, 0, 0]
        self.nosmooth = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(path)

    def load_model(self, checkpoint_path):
        model = wav2lip_mdoel()
        print("Load checkpoint from: {}".format(checkpoint_path))
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()
```



### ER-NeRF（Comming Soon）

ER-NeRF（ICCV2023）是使用最新的NeRF技术构建的数字人，拥有定制数字人的特性，只需要一个人的五分钟左右到视频即可重建出来，具体可参考 [https://github.com/Fictionarry/ER-NeRF](https://github.com/Fictionarry/ER-NeRF)

后续会针对此更新



## LLM - Conversation

### Linly-AI

Linly来自深圳大学数据工程国家重点实验室,参考[https://github.com/CVI-SZU/Linly](https://github.com/CVI-SZU/Linly)

下载Linly模型:[https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf](https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf)

可以使用`git`下载

```bash
git lfs install
git clone https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf
```

或者使用`huggingface`的下载工具`huggingface-cli`

```bash
pip install -U huggingface_hub

# 设置镜像加速
# Linux
export HF_ENDPOINT="https://hf-mirror.com"
# windows powershell
$env:HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download --resume-download Linly-AI/Chinese-LLaMA-2-7B-hf --local-dir Linly-AI/Chinese-LLaMA-2-7B-hf
```

或使用API:

```bash
# 命令行
curl -X POST -H "Content-Type: application/json" -d '{"question": "北京有什么好玩的地方?"}' http://url:port

# Python
import requests

url = "http://url:port"
headers = {
  "Content-Type": "application/json"
}

data = {
  "question": "北京有什么好玩的地方?" 
}

response = requests.post(url, headers=headers, json=data)
# response_text = response.content.decode("utf-8")
answer, tag = response.json()
# print(answer)
if tag == 'success':
    response_text =  answer[0]
else:
    print("fail")
print(response_text)
```

API部署推荐**FastAPI**，现在更新了 FastAPI 的API使用版本，FastAPI 是一个高性能、易用且现代的Python Web 框架，它通过使用最新的Python 特性和异步编程，提供了快速开发Web API 的能力。 该框架不仅易于学习和使用，还具有自动生成文档、数据验证等强大功能。 无论是构建小型项目还是大型应用程序，FastAPI 都是一个强大而有效的工具。

首先安装部署API所使用的库

```bash
pip install fastapi==0.104.1
pip install uvicorn==0.24.0.post1
```

其他使用方法大致相同，主要是不同代码实现方式，会更加简单边界，并且处理并发也会更好

```python
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch
from configs import model_path, api_port
# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的提示
    history = json_post_list.get('history')  # 获取请求中的历史记录
    max_length = json_post_list.get('max_length')  # 获取请求中的最大长度
    top_p = json_post_list.get('top_p')  # 获取请求中的top_p参数
    temperature = json_post_list.get('temperature')  # 获取请求中的温度参数
    
    # 调用模型进行对话生成
    prompt = f"请用少于25个字回答以下问题 ### Instruction:{prompt}  ### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(inputs.input_ids, 
                                  max_new_tokens=max_length if max_length else 2048,
                                  do_sample=True, 
                                  top_k=20,
                                  top_p=top_p,
                                  temperature=temperature if temperature else 0.84,
                                  repetition_penalty=1.15, eos_token_id=2, bos_token_id=1,pad_token_id=0)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = response.split("### Response:")[-1]
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        # "history": history,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0",
                                                    torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model.eval()  # 设置模型为评估模式
    # 启动FastAPI应用
    uvicorn.run(app, host='0.0.0.0', port=api_port, workers=1)  # 在指定端口和主机上启动应用
```

默认部署在 7871 端口，通过 POST 方法进行调用，可以使用curl调用，如下所示：

```bash
curl -X POST "http://127.0.0.1:7871" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "如何应对压力"}'
```

也可以使用python中的requests库进行调用，如下所示：

```python
import requests
import json

def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt}
    response = requests.post(url='http://127.0.0.1:7871', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == '__main__':
    print(get_completion('你好如何应对压力'))
```

得到的返回值如下所示：

```bash
{
  "response":"寻求支持和放松，并采取积极的措施解决问题。",
  "status":200,
  "time":"2024-01-12 01:43:37"
}
```



### Qwen

来自阿里云的Qwen，查看 [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)

如果想要快速使用，可以选1.8B的模型，参数比较少，在较小的显存也可以正常使用，当然这一部分可以替换

下载 Qwen1.8B 模型: [https://huggingface.co/Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)

可以使用`git`下载

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat
```

或者使用`huggingface`的下载工具`huggingface-cli`

```bash
pip install -U huggingface_hub

# 设置镜像加速
# Linux
export HF_ENDPOINT="https://hf-mirror.com"
# windows powershell
$env:HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download --resume-download Qwen/Qwen-1_8B-Chat --local-dir Qwen/Qwen-1_8B-Chat
```

如果出现了一些网络问题，大家其实可以用魔搭社区进行下载，速度很快，最后修改路径即可 [https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/files](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat/files)

```python
# 模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-1_8B-Chat')
```

### Gemini-Pro

来自 Google 的 Gemini-Pro，了解更多请访问 [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)

请求 API 密钥: [https://makersuite.google.com/](https://makersuite.google.com/)



### LLM 模型选择

在 app.py 文件中，轻松选择您需要的模型。

```python
# 可以注释掉选择模型
# llm = LLM(mode='offline').init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf')
# llm = LLM(mode='offline').init_model('Gemini', 'gemini-pro', api_key = "your api key")
# llm = LLM(mode='offline').init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')

# 可以通过config来设置模型
llm = LLM(mode=mode).init_model('Qwen', model_path)
```



## 优化

一些优化:

- 使用固定的输入人脸图像,提前提取特征,避免每次读取
- 移除不必要的库,缩短总时间
- 只保存最终视频输出,不保存中间结果,提高性能
- 使用OpenCV生成最终视频,比mimwrite更快

## Gradio

Gradio是一个Python库,提供了一种简单的方式将机器学习模型作为交互式Web应用程序来部署。

对Linly-Talker而言,使用Gradio有两个主要目的:

1. **可视化与演示**:Gradio为模型提供一个简单的Web GUI,上传图片和文本后可以直观地看到结果。这是展示系统能力的有效方式。

2. **用户交互**:Gradio的GUI可以作为前端,允许用户与Linly-Talker进行交互对话。用户可以上传自己的图片并输入问题,实时获取回答。这提供了更自然的语音交互方式。

具体来说,我们在app.py中创建了一个Gradio的Interface,接收图片和文本输入,调用函数生成回应视频,在GUI中显示出来。这样就实现了浏览器交互而不需要编写复杂的前端。

总之,Gradio为Linly-Talker提供了可视化和用户交互的接口,是展示系统功能和让最终用户使用系统的有效途径。

## 启动WebUI

之前我将很多个版本都是分开来的，实际上运行多个会比较麻烦，所以后续我增加了变成WebUI一个界面即可体验，后续也会不断更新

现在已加入WebUI的功能如下

- [x] 文本/语音数字人对话（固定数字人，分男女角色）
- [x] 任意图片数字人对话（可上传任意数字人）
- [x] 多轮GPT对话（加入历史对话数据，链接上下文）
- [x] 语音克隆对话（基于GPT-SoVITS设置进行语音克隆，内置烟嗓音，可根据语音对话的声音进行克隆）

```bash
# WebUI
python webui.py
```

![](docs/WebUI.png)



现在的启动一共有几种模式，可以选择特定的场景进行设置

第一种只有固定了人物问答，设置好了人物，省去了预处理时间

```bash
python app.py
```

![](docs/UI.png)

最近更新了第一种模式，加入了Wav2Lip模型进行对话

```bash
python appv2.py
```

第二种是可以任意上传图片进行对话

```bash
python app_img.py
```

![](docs/UI2.png)

第三种是在第一种的基础上加入了大语言模型，加入了多轮的GPT对话

```bash
python app_multi.py
```

![](docs/UI3.png)

现在加入了语音克隆的部分，可以自由切换自己克隆的声音模型和对应的人图片进行实现，这里我选择了一个烟嗓音和男生图片

```bash
python app_vits.py
```

文件夹结构如下

权重部分可以从这下载：[Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)

```bash
Linly-Talker/ 
├── app.py
├── app_img.py
├── utils.py
├── Linly-api.py
├── Linly-api-fast.py
├── Linly-example.ipynb
├── README.md
├── README_zh.md
├── request-Linly-api.py
├── requirements_app.txt
├── scripts
│   └── download_models.sh
├──	src
│   ├── audio2exp_models
│   ├── audio2pose_models
│   ├── config
│   ├── cost_time.py
│   ├── face3d
│   ├── facerender
│   ├── generate_batch.py
│   ├── generate_facerender_batch.py
│   ├── Record.py
│   ├── test_audio2coeff.py
│   └── utils
├── inputs
│   ├── example.png
│   └── first_frame_dir
│       ├── example_landmarks.txt
│       ├── example.mat
│       └── example.png
├── examples
│   └── source_image
│       ├── art_0.png
│       ├── ......
│       └── sad.png
├── TFG
│   ├── __init__.py
│   ├── Wav2Lip.py
│   └── SadTalker.py
└── TTS
│   ├── __init__.py
│   ├── EdgeTTS.py
│   └── TTS_app.py
├── ASR
│   ├── __init__.py
│   ├── FunASR.py
│   └── Whisper.py
├── LLM
│   ├── __init__.py
│   ├── Gemini.py
│   ├── Linly.py
│   └── Qwen.py
....... // 以下是需要下载的权重路径（可选）
├── checkpoints // SadTalker 权重路径
│   ├── mapping_00109-model.pth.tar
│   ├── mapping_00229-model.pth.tar
│   ├── SadTalker_V0.0.2_256.safetensors
│   └── SadTalker_V0.0.2_512.safetensors
│   ├── lipsync_expert.pth
│   ├── visual_quality_disc.pth
│   ├── wav2lip_gan.pth
│   └── wav2lip.pth // Wav2Lip 权重陆军
├── gfpgan // GFPGAN 权重路径
│   └── weights
│       ├── alignment_WFLW_4HG.pth
│       └── detection_Resnet50_Final.pth
├── Linly-AI // Linly 权重路径
│   └── Chinese-LLaMA-2-7B-hf 
│       ├── config.json
│       ├── generation_config.json
│       ├── pytorch_model-00001-of-00002.bin
│       ├── pytorch_model-00002-of-00002.bin
│       ├── pytorch_model.bin.index.json
│       ├── README.md
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── tokenizer.model
├── Qwen // Qwen 权重路径
│   └── Qwen-1_8B-Chat
│       ├── cache_autogptq_cuda_256.cpp
│       ├── cache_autogptq_cuda_kernel_256.cu
│       ├── config.json
│       ├── configuration_qwen.py
│       ├── cpp_kernels.py
│       ├── examples
│       │   └── react_prompt.md
│       ├── generation_config.json
│       ├── LICENSE
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── modeling_qwen.py
│       ├── model.safetensors.index.json
│       ├── NOTICE
│       ├── qwen_generation_utils.py
│       ├── qwen.tiktoken
│       ├── README.md
│       ├── tokenization_qwen.py
│       └── tokenizer_config.json
```



## 参考

- [https://github.com/openai/whisper](https://github.com/openai/whisper)
- [https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)  
- [https://github.com/CVI-SZU/Linly](https://github.com/CVI-SZU/Linly)
- [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)
- [https://github.com/OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker)
- [https://github.com/RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kedreamix/Linly-Talker&type=Date)](https://star-history.com/#Kedreamix/Linly-Talker&Date)

