# Linly-Talker

[English](./README.md) [简体中文](./README_zh.md)

## 介绍

Linly-Talker是一个将大型语言模型与视觉模型相结合的智能AI系统,创建了一种全新的人机交互方式。它集成了各种技术,例如Whisper、Linly、微软语音服务和SadTalker会说话的生成系统。该系统部署在Gradio上,允许用户通过提供图像与AI助手进行交谈。用户可以根据自己的喜好进行自由的对话或内容生成。

![The system architecture of multimodal human–computer interaction.](HOI.png)

## 创建环境

```
conda create -n linly python=3.8 
conda activate linly

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

conda install ffmpeg 

pip install -r requirements_app.txt
```

## ASR - Whisper

借鉴OpenAI的Whisper,具体使用方法参考https://github.com/openai/whisper

## TTS - Edge TTS

使用微软语音服务,具体使用方法参考https://github.com/rany2/edge-tts

## THG - SadTalker 

说话头生成使用SadTalker,参考CVPR 2023,详情见https://sadtalker.github.io

下载SadTalker模型:

```
bash scripts/download_models.sh  
```

## LLM - Linly

Linly来自深圳大学数据工程国家重点实验室,参考https://github.com/CVI-SZU/Linly

下载Linly模型:https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf

```
git lfs install
git clone https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf
```

或使用API:

```
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

```
python app.py
```

![](UI.jpg)


## 参考

- https://github.com/openai/whisper
- https://github.com/rany2/edge-tts  
- https://github.com/CVI-SZU/Linly
- https://github.com/OpenTalker/SadTalker
