# Digital Human Intelligent Dialogue System - Linly-Talker — 'Interactive Dialogue with Your Virtual Self'

<div align="center">
<h1>Linly-Talker WebUI</h1>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/Kedreamix/Linly-Talker)

<img src="docs/linly_logo.png" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/Kedreamix/Linly-Talker/blob/main/colab_webui.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/Kedreamix/Linly-Talker/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models%20Repo-yellow.svg?style=for-the-badge)](https://huggingface.co/Kedreamix/Linly-Talker)

[**English**](./README.md) | [**中文简体**](./README_zh.md)

</div>

**2023.12 Update** 📆

**Users can upload any images for the conversation**

**2024.01 Update** 📆📆

- **Exciting news! I've now incorporated both the powerful GeminiPro and Qwen large models into our conversational scene. Users can now upload images during the conversation, adding a whole new dimension to the interactions.** 
- **The deployment invocation method for FastAPI has been updated.**
- **The advanced settings options for Microsoft TTS have been updated, increasing the variety of voice types. Additionally, video subtitles have been introduced to enhance visualization.**
- **Updated the GPT multi-turn conversation system to establish contextual connections in dialogue, enhancing the interactivity and realism of the digital persona.**

**2024.02 Update** 📆

- **Updated Gradio to the latest version 4.16.0, providing the interface with additional functionalities such as capturing images from the camera to create digital personas, among others.**
- **ASR and THG have been updated. FunASR from Alibaba has been integrated into ASR, enhancing its speed significantly. Additionally, the THG section now incorporates the Wav2Lip model, while ER-NeRF is currently in preparation (Coming Soon).**
- **I have incorporated the GPT-SoVITS model, which is a voice cloning method. By fine-tuning it with just one minute of a person's speech data, it can effectively clone their voice. The results are quite impressive and worth recommending.**
- **I have integrated a web user interface (WebUI) that allows for better execution of Linly-Talker.**

**2024.04 Update** 📆

- **Updated the offline mode for Paddle TTS, excluding Edge TTS.**
- **Updated ER-NeRF as one of the choices for Avatar generation.**
- **Updated app_talk.py to allow for the free upload of voice and images/videos for generation without being based on a dialogue scenario.**

---

<details>
<summary>Content</summary>

<!-- TOC -->

- [Digital Avatar Conversational System - Linly-Talker —— "Digital Persona Interaction: Interact with Your Virtual Self”](#digital-avatar-conversational-system---linly-talker--digital-persona-interaction-interact-with-your-virtual-self)
  - [Introduction](#introduction)
  - [TO DO LIST](#to-do-list)
  - [Example](#example)
  - [Setup Environment](#setup-environment)
  - [ASR - Speech Recognition](#asr---speech-recognition)
    - [Whisper](#whisper)
    - [FunASR](#funasr)
  - [TTS - Edge TTS](#tts---edge-tts)
  - [Voice Clone](#voice-clone)
    - [GPT-SoVITS（Recommend）](#gpt-sovitsrecommend)
    - [XTTS](#xtts)
  - [THG - Avatar](#thg---avatar)
    - [SadTalker](#sadtalker)
    - [Wav2Lip](#wav2lip)
    - [ER-NeRF (Coming Soon)](#er-nerf-coming-soon)
  - [LLM - Conversation](#llm---conversation)
    - [Linly-AI](#linly-ai)
    - [Qwen](#qwen)
    - [Gemini-Pro](#gemini-pro)
    - [LLM Model Selection](#llm-model-selection)
  - [Optimizations](#optimizations)
  - [Gradio](#gradio)
  - [Start WebUI](#start-webui)
  - [Folder structure](#folder-structure)
  - [Reference](#reference)
  - [Star History](#star-history)

<!-- /TOC -->

</details>


## Introduction

Linly-Talker is an innovative digital human conversation system that integrates the latest artificial intelligence technologies, including Large Language Models (LLM) 🤖, Automatic Speech Recognition (ASR) 🎙️, Text-to-Speech (TTS) 🗣️, and voice cloning technology 🎤. This system offers an interactive web interface through the Gradio platform 🌐, allowing users to upload images 📷 and engage in personalized dialogues with AI 💬.

The core features of the system include:

1. **Multi-Model Integration**: Linly-Talker combines major models such as Linly, GeminiPro, Qwen, as well as visual models like Whisper, SadTalker, to achieve high-quality dialogues and visual generation.
2. **Multi-Turn Conversational Ability**: Through the multi-turn dialogue system powered by GPT models, Linly-Talker can understand and maintain contextually relevant and coherent conversations, significantly enhancing the authenticity of the interaction.
3. **Voice Cloning**: Utilizing technologies like GPT-SoVITS, users can upload a one-minute voice sample for fine-tuning, and the system will clone the user's voice, enabling the digital human to converse in the user's voice.
4. **Real-Time Interaction**: The system supports real-time speech recognition and video captioning, allowing users to communicate naturally with the digital human via voice.
5. **Visual Enhancement**: With digital human generation technologies, Linly-Talker can create realistic digital human avatars, providing a more immersive experience.

The design philosophy of Linly-Talker is to create a new form of human-computer interaction that goes beyond simple Q&A. By integrating advanced technologies, it offers an intelligent digital human capable of understanding, responding to, and simulating human communication.

![The system architecture of multimodal human–computer interaction.](docs/HOI_en.png)

> You can watch the demo video [here](https://www.bilibili.com/video/BV1rN4y1a76x/).
>
> I have recorded a series of videos on Bilibili, which also represent every step of my updates and methods of use. For detailed information, please refer to [Digital Human Dialogue System - Linly-Talker Collection](https://space.bilibili.com/241286257/channel/collectiondetail?sid=2065753).
>
> - [🔥🔥🔥 Digital Human Dialogue System Linly-Talker 🔥🔥🔥](https://www.bilibili.com/video/BV1rN4y1a76x/)
> - [🚀 The Future of Digital Humans: The Empowerment Path of Linly-Talker + GPT-SoVIT Voice Cloning Technology](https://www.bilibili.com/video/BV1S4421A7gh/)
> - [Deploying Linly-Talker on AutoDL Platform (Super Detailed Tutorial for Beginners)](https://www.bilibili.com/video/BV1uT421m74z/)
> - [Linly-Talker Update: Offline TTS Integration and Customized Digital Human Solutions](https://www.bilibili.com/video/BV1Mr421u7NN/)

## TO DO LIST

- [x] Completed the basic conversation system flow, capable of `voice interactions`.
- [x] Integrated the LLM large model, including the usage of `Linly`, `Qwen`, and `GeminiPro`.
- [x] Enabled the ability to upload `any digital person's photo` for conversation.
- [x] Integrated `FastAPI` invocation for Linly.
- [x] Utilized Microsoft `TTS` with advanced options, allowing customization of voice and tone parameters to enhance audio diversity.
- [x] `Added subtitles` to video generation for improved visualization.
- [x] GPT `Multi-turn Dialogue System` (Enhance the interactivity and realism of digital entities, bolstering their intelligence)
- [x] Optimized the Gradio interface by incorporating additional models such as Wav2Lip, FunASR, and others.
- [x] `Voice Cloning` Technology (Synthesize one's own voice using voice cloning to enhance the realism and interactive experience of digital entities)
- [x] Integrate offline TTS (Text-to-Speech) along with NeRF-based methods and models.
- [ ] `Real-time` Speech Recognition (Enable conversation and communication between humans and digital entities using voice)

🔆 The Linly-Talker project is ongoing - pull requests are welcome! If you have any suggestions regarding new model approaches, research, techniques, or if you discover any runtime errors, please feel free to edit and submit a pull request. You can also open an issue or contact me directly via email. 📩⭐ If you find this repository useful, please give it a star! 🤩

> If you encounter any issues during deployment, please consult the [Common Issues Summary](https://github.com/Kedreamix/Linly-Talker/blob/main/常见问题汇总.md) section, where I have compiled a list of all potential problems. Additionally, a discussion group is available here, and I will provide regular updates. Thank you for your attention and use of Linly-Talker!

## Example

|                        文字/语音对话                         |                          数字人回答                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                 应对压力最有效的方法是什么？                 | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/f1deb189-b682-4175-9dea-7eeb0fb392ca"></video> |
|                      如何进行时间管理？                      | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/968b5c43-4dce-484b-b6c6-0fd4d621ac03"></video> |
|  撰写一篇交响乐音乐会评论，讨论乐团的表演和观众的整体体验。  | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/f052820f-6511-4cf0-a383-daf8402630db"></video> |
| 翻译成中文：Luck is a dividend of sweat. The more you sweat, the luckier you get. | <video src="https://github.com/Kedreamix/Linly-Talker/assets/61195303/118eec13-a9f7-4c38-b4ad-044d36ba9776"></video> |

## Setup Environment

To install the environment using Anaconda and PyTorch, follow the steps below:

```bash
conda create -n linly python=3.10
conda activate linly

# PyTorch Installation Method 1: Conda Installation (Recommended)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# PyTorch Installation Method 2: Pip Installation
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

conda install -q ffmpeg # ffmpeg==4.2.2

pip install -r requirements_app.txt
```

If you want to use models like voice cloning, you may need a higher version of PyTorch. However, the functionality will be more diverse. You may need to use CUDA 11.8 as the driver version, which you can choose.

```bash
conda create -n linly python=3.10  
conda activate linly

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

conda install -q ffmpeg # ffmpeg==4.2.2

pip install -r requirements_app.txt

# Install dependencies for voice cloning
pip install -r VITS/requirements_gptsovits.txt
```

If you wish to use NeRF-based models, you may need to set up the corresponding environment:

```bash
# Install dependencies for NeRF
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r TFG/requirements_nerf.txt

# If there are issues with PyAudio, you can install the corresponding dependencies
# sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0

# Note the following modules. If installation is unsuccessful, you can navigate to the path and use pip install . or python setup.py install to compile and install.
# NeRF/freqencoder
# NeRF/gridencoder
# NeRF/raymarching
# NeRF/shencoder
```

If you are using PaddleTTS, you can set up the corresponding environment with:

```bash
pip install -r TTS/requirements_paddle.txt
```

Next, you need to install the corresponding models. You can download them using the following methods. Once downloaded, place the files in the specified folder structure (explained at the end of this document).

- [Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)
- [huggingface](https://huggingface.co/Kedreamix/Linly-Talker)
- [modelscope](https://www.modelscope.cn/models/Kedreamix/Linly-Talker/summary)

**HuggingFace Download**

If the download speed is too slow, consider using a mirror site. For more information, refer to [Efficiently Obtain Hugging Face Models Using Mirror Sites](https://kedreamix.github.io/2024/01/05/Note/HuggingFace/?highlight=镜像).

```bash
# Download pre-trained models from Hugging Face
git lfs install
git clone https://huggingface.co/Kedreamix/Linly-Talker
```

**ModelScope Download**

```bash
# Download pre-trained models from ModelScope
# 1. Git method
git lfs install
git clone https://www.modelscope.cn/Kedreamix/Linly-Talker.git

# 2. Python code download
pip install modelscope
from modelscope import snapshot_download
model_dir = snapshot_download('Kedreamix/Linly-Talker')
```

**Move All Models to the Current Directory**

If you downloaded from Baidu Netdisk, you can refer to the directory structure at the end of the document to move the models.

```bash
# Move all models to the current directory
# Checkpoints contain SadTalker and Wav2Lip
mv Linly-Talker/checkpoints/* ./checkpoints

# Enhanced GFPGAN for SadTalker
# pip install gfpgan
# mv Linly-Talker/gfpan ./

# Voice cloning models
mv Linly-Talker/GPT_SoVITS/pretrained_models/* ./GPT_SoVITS/pretrained_models/

# Qwen large model
mv Linly-Talker/Qwen ./
```

For the convenience of deployment and usage, an `configs.py` file has been updated. You can modify some hyperparameters in this file for customization:

```bash
# Device Running Port
port = 7870

# API Running Port and IP
# Localhost port is 127.0.0.1; for global port forwarding, use "0.0.0.0"
ip = '127.0.0.1'
api_port = 7871

# Linly Model Path
mode = 'api'  # For 'api', Linly-api-fast.py must be run first
mode = 'offline'
model_path = 'Linly-AI/Chinese-LLaMA-2-7B-hf'

# SSL Certificate (required for microphone interaction)
# Preferably an absolute path
ssl_certfile = "./https_cert/cert.pem"
ssl_keyfile = "./https_cert/key.pem"
```

This file allows you to adjust parameters such as the device running port, API running port, Linly model path, and SSL certificate paths for ease of deployment and configuration.

## ASR - Speech Recognition

For detailed information about the usage and code implementation of Automatic Speech Recognition (ASR), please refer to [ASR - Bridging the Gap with Digital Humans](./ASR/README.md).

### Whisper

To implement ASR (Automatic Speech Recognition) using OpenAI's Whisper, you can refer to the specific usage methods provided in the GitHub repository: [https://github.com/openai/whisper](https://github.com/openai/whisper)



### FunASR

The speech recognition performance of Alibaba's FunASR is quite impressive and it is actually better than Whisper in terms of Chinese language. Additionally, FunASR is capable of achieving real-time results, making it a great choice. You can experience FunASR by accessing the FunASR file in the ASR folder. Please refer to [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR) for more information.



## TTS - Text To Speech

For detailed information about the usage and code implementation of Text-to-Speech (TTS), please refer to [TTS - Empowering Digital Humans with Natural Speech Interaction](./TTS/README.md).

### Edge TTS

To use Microsoft Edge's online text-to-speech service from Python without needing Microsoft Edge or Windows or an API key, you can refer to the GitHub repository at [https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts). It provides a Python module called "edge-tts" that allows you to utilize the service. You can find detailed installation instructions and usage examples in the repository's README file.

### PaddleTTS

In practical use, there may be scenarios that require offline operation. Since Edge TTS requires an online environment to generate speech, we have chosen PaddleSpeech, another open-source alternative, for Text-to-Speech (TTS). Although there might be some differences in the quality, PaddleSpeech supports offline operations. For more information, you can refer to the GitHub page of PaddleSpeech: [https://github.com/PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech).



## Voice Clone

For detailed information about the usage and code implementation of Voice Clone, please refer to [Voice Clone - Stealing Your Voice Quietly During Conversations](./VITS/README.md).

### GPT-SoVITS（Recommend）

Thank you for your open source contribution. I have also found the `GPT-SoVITS` voice cloning model to be quite impressive. You can find the project at [https://github.com/RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).



### XTTS

Coqui XTTS is a leading deep learning toolkit for Text-to-Speech (TTS) tasks, allowing for voice cloning and voice transfer to different languages using a 5-second or longer audio clip.

🐸 TTS is a library for advanced text-to-speech generation.

🚀 Over 1100 pre-trained models for various languages.

🛠️ Tools for training new models and fine-tuning existing models in any language.

📚 Utility programs for dataset analysis and management.

- Experience XTTS online [https://huggingface.co/spaces/coqui/xtts](https://huggingface.co/spaces/coqui/xtts)
- Official GitHub repository: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)



## THG - Avatar

Detailed information about the usage and code implementation of digital human generation can be found in [THG - Building Intelligent Digital Humans](./TFG/README.md).

### SadTalker

Digital persona generation can utilize SadTalker (CVPR 2023). For detailed information, please visit [https://sadtalker.github.io](https://sadtalker.github.io).

Before usage, download the SadTalker model:

```bash
bash scripts/sadtalker_download_models.sh  
```

[Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)

> If downloading from Baidu Cloud, remember to place it in the `checkpoints` folder. The model downloaded from Baidu Cloud is named `sadtalker` by default, but it should be renamed to `checkpoints`.

### Wav2Lip

Digital persona generation can also utilize Wav2Lip (ACM 2020). For detailed information, refer to [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip).

Before usage, download the Wav2Lip model:

| Model                        | Description                                           | Link to the model                                            |
| ---------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Wav2Lip                      | Highly accurate lip-sync                              | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW) |
| Wav2Lip + GAN                | Slightly inferior lip-sync, but better visual quality | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |
| Expert Discriminator         | Weights of the expert discriminator                   | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP) |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup     | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo) |



### ER-NeRF (Coming Soon)

ER-NeRF (ICCV 2023) is a digital human built using the latest NeRF technology. It allows for the customization of digital characters and can reconstruct them using just a five-minute video of a person. For more details, please refer to [https://github.com/Fictionarry/ER-NeRF](https://github.com/Fictionarry/ER-NeRF).

Updates have been made in the app_talk.py section. If better results are desired, it might be considered to clone and customize the voice of a digital human to achieve improved effects.



## LLM - Conversation

For detailed information about the usage and code implementation of Large Language Models (LLM), please refer to [LLM - Empowering Digital Humans with Powerful Language Models](./LLM/README.md).

### Linly-AI

Linly-AI is a Large Language model developed by CVI at Shenzhen University. You can find more information about Linly-AI on their GitHub repository: https://github.com/CVI-SZU/Linly

Download Linly models: [https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf](https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf)

You can use `git` to download:

```bash
git lfs install
git clone https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf
```

Alternatively, you can use the `huggingface` download tool `huggingface-cli`:

```bash
pip install -U huggingface_hub

# Set up mirror acceleration
# Linux
export HF_ENDPOINT="https://hf-mirror.com"
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download --resume-download Linly-AI/Chinese-LLaMA-2-7B-hf --local-dir Linly-AI/Chinese-LLaMA-2-7B-hf
```



### Qwen

Qwen is an AI model developed by Alibaba Cloud. You can check out the GitHub repository for Qwen here: https://github.com/QwenLM/Qwen

If you want to quickly use Qwen, you can choose the 1.8B model, which has fewer parameters and can run smoothly even with limited GPU memory. Of course, this part can be replaced with other options.

You can download the Qwen 1.8B model from this link: https://huggingface.co/Qwen/Qwen-1_8B-Chat

You can use `git` to download:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat
```

Alternatively, you can use the `huggingface` download tool `huggingface-cli`:

```bash
pip install -U huggingface_hub

# Set up mirror acceleration
# Linux
export HF_ENDPOINT="https://hf-mirror.com"
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download --resume-download Qwen/Qwen-1_8B-Chat --local-dir Qwen/Qwen-1_8B-Chat
```



### Gemini-Pro

Gemini-Pro is an AI model developed by Google. To learn more about Gemini-Pro, you can visit their website: https://deepmind.google/technologies/gemini/

If you want to request an API key for Gemini-Pro, you can visit this link: https://makersuite.google.com/



### LLM Model Selection

In the app.py file, tailor your model choice with ease.

```python
# Uncomment and set up the model of your choice:

# llm = LLM(mode='offline').init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf')
# llm = LLM(mode='offline').init_model('Gemini', 'gemini-pro', api_key = "your api key")
# llm = LLM(mode='offline').init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')

# Manual download with a specific path
llm = LLM(mode=mode).init_model('Qwen', model_path)
```



## Optimizations

Some optimizations:

- Use fixed input face images, extract features beforehand to avoid reading each time
- Remove unnecessary libraries to reduce total time
- Only save final video output, don't save intermediate results to improve performance 
- Use OpenCV to generate final video instead of mimwrite for faster runtime

## Gradio

Gradio is a Python library that provides an easy way to deploy machine learning models as interactive web apps. 

For Linly-Talker, Gradio serves two main purposes:

1. **Visualization & Demo**: Gradio provides a simple web GUI for the model, allowing users to see the results intuitively by uploading an image and entering text. This is an effective way to showcase the capabilities of the system.

2. **User Interaction**: The Gradio GUI can serve as a frontend to allow end users to interact with Linly-Talker. Users can upload their own images and ask arbitrary questions or have conversations to get real-time responses. This provides a more natural speech interaction method.

Specifically, we create a Gradio Interface in app.py that takes image and text inputs, calls our function to generate the response video, and displays it in the GUI. This enables browser interaction without needing to build complex frontend. 

In summary, Gradio provides visualization and user interaction interfaces for Linly-Talker, serving as effective means for showcasing system capabilities and enabling end users.

## Start WebUI

Previously, I had separated many versions, but it became cumbersome to run multiple versions. Therefore, I have added a WebUI feature to provide a single interface for a seamless experience. I will continue to update it in the future.

The current features available in the WebUI are as follows:

- [x] Text/Voice-based dialogue with virtual characters (fixed characters with male and female roles)
- [x] Dialogue with virtual characters using any image (you can upload any character image)
- [x] Multi-turn GPT dialogue (incorporating historical dialogue data to maintain context)
- [x] Voice cloning dialogue (based on GPT-SoVITS settings for voice cloning, including a built-in smoky voice that can be cloned based on the voice of the dialogue)

```bash
# WebUI
python webui.py
```

![](docs/WebUI.png)

There are three modes for the current startup, and you can choose a specific setting based on the scenario.

The first mode involves fixed Q&A with a predefined character, eliminating preprocessing time.

```bash
python app.py
```

![](docs/UI.png)

The first mode has recently been updated to include the Wav2Lip model for dialogue.

```bash
python appv2.py
```



The second mode allows for conversing with any uploaded image.

```bash
python app_img.py
```

![](docs/UI2.png)

The third mode builds upon the first one by incorporating a large language model for multi-turn GPT conversations.

```bash
python app_multi.py
```

![](docs/UI3.png)

Now, the part of voice cloning has been added, allowing for freely switching between cloned voice models and corresponding person images. Here, I have chosen a deep, smoky voice and an image of a male.

```bash
python app_vits.py
```

A fourth method has been added, which does not fixate on a specific scenario for conversation. Instead, it allows for direct input of voice or the generation of voice for the creation of a digital human. It incorporates methods such as Sadtalker, Wav2Lip, and ER-NeRF.

> ER-NeRF is trained on videos of a single individual, so a specific model needs to be replaced to render and obtain the correct results. It comes with pre-installed weights for Obama, which can be used directly with the following command:

```bash
python app_talk.py
```

![](docs/UI4.png)



## Folder structure

The folder structure of the weight files is as follows:

- `Baidu (百度云盘)`: You can download the weights from [here](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`).
- `huggingface`: You can access the weights at [this link](https://huggingface.co/Kedreamix/Linly-Talker).
- `modelscope`: The weights will be available soon at [this link](https://www.modelscope.cn/models/Kedreamix/Linly-Talker/files).

```bash
Linly-Talker/ 
├── checkpoints
│   ├── hub
│   │   └── checkpoints
│   │       └── s3fd-619a316812.pth
│   ├── lipsync_expert.pth
│   ├── mapping_00109-model.pth.tar
│   ├── mapping_00229-model.pth.tar
│   ├── SadTalker_V0.0.2_256.safetensors
│   ├── visual_quality_disc.pth
│   ├── wav2lip_gan.pth
│   └── wav2lip.pth
├── gfpgan
│   └── weights
│       ├── alignment_WFLW_4HG.pth
│       └── detection_Resnet50_Final.pth
├── GPT_SoVITS
│   └── pretrained_models
│       ├── chinese-hubert-base
│       │   ├── config.json
│       │   ├── preprocessor_config.json
│       │   └── pytorch_model.bin
│       ├── chinese-roberta-wwm-ext-large
│       │   ├── config.json
│       │   ├── pytorch_model.bin
│       │   └── tokenizer.json
│       ├── README.md
│       ├── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
│       ├── s2D488k.pth
│       ├── s2G488k.pth
│       └── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
├── Qwen
│   └── Qwen-1_8B-Chat
│       ├── assets
│       │   ├── logo.jpg
│       │   ├── qwen_tokenizer.png
│       │   ├── react_showcase_001.png
│       │   ├── react_showcase_002.png
│       │   └── wechat.png
│       ├── cache_autogptq_cuda_256.cpp
│       ├── cache_autogptq_cuda_kernel_256.cu
│       ├── config.json
│       ├── configuration_qwen.py
│       ├── cpp_kernels.py
│       ├── examples
│       │   └── react_prompt.md
│       ├── generation_config.json
│       ├── LICENSE
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── modeling_qwen.py
│       ├── model.safetensors.index.json
│       ├── NOTICE
│       ├── qwen_generation_utils.py
│       ├── qwen.tiktoken
│       ├── README.md
│       ├── tokenization_qwen.py
│       └── tokenizer_config.json
└── README.md
```



## Reference

**ASR**

- [https://github.com/openai/whisper](https://github.com/openai/whisper)
- [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR)

**TTS**

- [https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)  

**LLM**

- [https://github.com/CVI-SZU/Linly](https://github.com/CVI-SZU/Linly)
- [https://github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)

**THG**

- [https://github.com/OpenTalker/SadTalker](https://github.com/OpenTalker/SadTalker)
- [https://github.com/Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip)

**Voice Clone**

- [https://github.com/RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kedreamix/Linly-Talker&type=Date)](https://star-history.com/#Kedreamix/Linly-Talker&Date)

