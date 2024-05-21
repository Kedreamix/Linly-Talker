## LLM 大语言模型为数字人赋能



### Linly-AI 伶荔

Linly来自深圳大学数据工程国家重点实验室，参考[https://github.com/CVI-SZU/Linly](https://github.com/CVI-SZU/Linly)

下载Linly模型：[https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf](https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf)

一共有两种下载方式：

1. 可以使用`git`下载

```bash
git lfs install
git clone https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf
```

2. 使用`huggingface`的下载工具`huggingface-cli`

```bash
pip install -U huggingface_hub

# 设置镜像加速
# Linux
export HF_ENDPOINT="https://hf-mirror.com"
# windows powershell
$env:HF_ENDPOINT="https://hf-mirror.com"

# 命令行下载
huggingface-cli download --resume-download Linly-AI/Chinese-LLaMA-2-7B-hf --local-dir Linly-AI/Chinese-LLaMA-2-7B-hf
```



**API部署**

API部署推荐**FastAPI**，现在更新了 FastAPI 的API使用版本，FastAPI 是一个高性能、易用且现代的Python Web 框架，它通过使用最新的Python 特性和异步编程，提供了快速开发Web API 的能力。 该框架不仅易于学习和使用，还具有自动生成文档、数据验证等强大功能。 无论是构建小型项目还是大型应用程序，FastAPI 都是一个强大而有效的工具。

首先安装部署API所使用的库

```bash
pip install fastapi==0.104.1
pip install uvicorn==0.24.0.post1
```

其他使用方法大致相同，主要是不同代码实现方式，会更加简单便捷，并且处理并发也会更好

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

**POST调用**

默认部署在 7871 端口，通过 POST 方法进行调用，可以使用curl调用，如下所示：

```bash
curl -X POST "http://127.0.0.1:7871" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "如何应对压力"}'
```

**Python代码调用**

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



### Qwen 通义千问

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



### Gemini-Pro 双子座

来自 Google 的 Gemini-Pro，了解更多请访问 [https://deepmind.google/technologies/gemini/](https://deepmind.google/technologies/gemini/)

请求 API 密钥: [https://makersuite.google.com/](https://makersuite.google.com/)



### ChatGPT

ChatGPT由OpenAI开发，需要申请API。更多信息请访问 [OpenAI平台文档](https://platform.openai.com/docs/introduction)。



### ChatGLM

ChatGLM由清华大学开发，更多信息请访问 [ChatGLM GitHub页面](https://github.com/THUDM/ChatGLM3)。



### 欢迎补充～～～
