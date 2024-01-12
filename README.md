# 数字人对话系统 - Linly-Talker —— “数字人交互，与虚拟的自己互动”

[English](./README_en.md) [简体中文](./README.md)

**2023.12 更新** 📆

**用户可以上传任意图片进行对话**

**2024.01 更新** 📆

- **令人兴奋的消息！我现在已经将强大的GeminiPro和Qwen大模型融入到我们的对话场景中。用户现在可以在对话中上传任何图片，为我们的互动增添了全新的层面。**
-  **更新了FastAPI的部署调用方法。** 
- **更新了微软TTS的高级设置选项，增加声音种类的多样性，以及加入视频字幕加强可视化。**

## 介绍

Linly-Talker是一个将大型语言模型与视觉模型相结合的智能AI系统,创建了一种全新的人机交互方式。它集成了各种技术,例如Whisper、Linly、微软语音服务和SadTalker会说话的生成系统。该系统部署在Gradio上,允许用户通过提供图像与AI助手进行交谈。用户可以根据自己的喜好进行自由的对话或内容生成。

![The system architecture of multimodal human–computer interaction.](docs/HOI.png)

## TO DO LIST

- [x] 基本完成对话系统流程，能够语音对话
- [x] 加入了LLM大模型，包括Linly，Qwen和GeminiPro的使用
- [x] 可上传任意数字人照片进行对话
- [x] Linly加入FastAPI调用方式
- [x] 利用微软TTS加入高级选项，可设置对应人声以及音调等参数，增加声音的多样性
- [x] 视频生成加入字幕，能够更好的进行可视化
- [ ] 语音克隆技术（语音克隆合成自己声音，提高数字人分身的真实感和互动体验）
- [ ] 实时语音识别（人与数字人之间就可以通过语音进行对话交流)
- [ ] GPT多轮对话系统（提高数字人的交互性和真实感，增强数字人的智能）

🔆 该项目 Linly-Talker 正在进行中 - 欢迎提出PR请求！如果您有任何关于新的模型方法、研究、技术或发现运行错误的建议，请随时编辑并提交 PR。您也可以打开一个问题或通过电子邮件直接联系我。📩⭐ 如果您发现这个Github Project有用，请给它点个星！🤩

## 创建环境

```bash
conda create -n linly python=3.8 
conda activate linly

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

conda install ffmpeg 

pip install -r requirements_app.txt
```

为了大家的部署使用方便，更新了一个`configs.py`文件，可以对其进行一些超参数修改即可

```bash
# 设备运行端口 (Device running port)
port = 7870
# api运行端口 (API running port)
api_port = 7871
# Linly模型路径 (Linly model path)
mode = 'api' # api 需要先运行Linly-api-fast.py
mode = 'offline'
model_path = 'Linly-AI/Chinese-LLaMA-2-7B-hf'
# ssl证书 (SSL certificate) 麦克风对话需要此参数
ssl_certfile = "/path/to/Linly-Talker/cert.pem"
ssl_keyfile = "/path/to/Linly-Talker/key.pem"
```

## ASR - Whisper

借鉴OpenAI的Whisper,具体使用方法参考[https://github.com/openai/whisper](https://github.com/openai/whisper)

## TTS - Edge TTS

使用微软语音服务,具体使用方法参考[https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)

## THG - SadTalker

说话头生成使用SadTalker（CVPR 2023）,详情见[https://sadtalker.github.io](https://sadtalker.github.io)

下载SadTalker模型:

```bash
bash scripts/download_models.sh  
```

## LLM - Conversation

### Linly-AI

Linly来自深圳大学数据工程国家重点实验室,参考[https://github.com/CVI-SZU/Linly](https://github.com/CVI-SZU/Linly)

下载Linly模型:[https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf](https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf)

```bash
git lfs install
git clone https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf
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

```bash
import requests
import json

def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt}
    response = requests.post(url='http://127.0.0.1:7871', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == '__main__':
    print(get_completion('你好如何应对压力
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

下载 Qwen 模型: [https://huggingface.co/Qwen/Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4)

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat
```



### Gemini-Pro

来自 Google 的 Gemini-Pro，了解更多请访问 [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)

请求 API 密钥: [https://makersuite.google.com/](https://makersuite.google.com/)



### 模型选择

在 app.py 文件中，轻松选择您需要的模型。

```python
# 取消注释并设置您选择的模型:

# llm = Gemini(model_path='gemini-pro', api_key=None, proxy_url=None) # 不要忘记加入您自己的 Google API 密钥
# llm = Qwen(mode='offline', model_path="Qwen/Qwen-1_8B-Chat")
# 自动下载
# llm = Linly(mode='offline', model_path="Linly-AI/Chinese-LLaMA-2-7B-hf")
# 手动下载到指定路径
llm = Linly(mode='offline', model_path="./Chinese-LLaMA-2-7B-hf")
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

## 启动

首先说明一下的文件夹结构如下

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
│	└── .....
├── inputs
│   ├── example.png
│   └── first_frame_dir
│       ├── example_landmarks.txt
│       ├── example.mat
│       └── example.png
├── examples
│   ├── driven_audio
│   │   ├── bus_chinese.wav
│   │   ├── ......
│   │   └── RD_Radio40_000.wav
│   ├── ref_video
│   │   ├── WDA_AlexandriaOcasioCortez_000.mp4
│   │   └── WDA_KatieHill_000.mp4
│   └── source_image
│       ├── art_0.png
│       ├── ......
│       └── sad.png
├── checkpoints // SadTalker 权重路径
│   ├── mapping_00109-model.pth.tar
│   ├── mapping_00229-model.pth.tar
│   ├── SadTalker_V0.0.2_256.safetensors
│   └── SadTalker_V0.0.2_512.safetensors
├── gfpgan // GFPGAN 权重路径
│   └── weights
│       ├── alignment_WFLW_4HG.pth
│       └── detection_Resnet50_Final.pth
├── Linly-AI
    ├── Chinese-LLaMA-2-7B-hf // Linly 权重路径
        ├── config.json
        ├── generation_config.json
        ├── pytorch_model-00001-of-00002.bin
        ├── pytorch_model-00002-of-00002.bin
        ├── pytorch_model.bin.index.json
        ├── README.md
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.model
```

接下来进行启动

```bash
python app.py
```

![](docs/UI.png)

可以任意上传图片进行对话

```bash
python app_img.py
```

![](docs/UI2.png)



## 参考

- [https://github.com/openai/whisper](https://github.com/openai/whisper)
- [https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)  
- [https://github.com/CVI-SZU/Linly](https://github.com/CVI-SZU/Linly)
- [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)
- [https://github.com/OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kedreamix/Linly-Talker&type=Date)](https://star-history.com/#Kedreamix/Linly-Talker&Date)

