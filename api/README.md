# Linly-Talker API 接口文档

## 0. 安装依赖

首先，安装所需的依赖库：

```bash
pip install -r api/requirements.txt
```

---

## 1. 文字转语音 (TTS) API

### 启动命令

要启动 TTS API 服务，有两种方式：

1. 使用 `fastapi` 命令：

    ```bash
    fastapi dev api/tts_api.py --host 0.0.0.0 --port 8001
    ```

2. 直接运行 Python 脚本：

    ```bash
    python api/tts_api.py
    ```

启动服务后，运行客户端脚本来测试 API：

```bash
python api/tts_client.py
```

生成的 Wav 音频文件会保存在 `output` 文件夹下。

### API 端点

- **基础 URL**: `http://localhost:8001`
- **API 文档**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **更换模型**: [http://localhost:8001/change_model](http://localhost:8001/change_model)
- **模型预测**: [http://localhost:8001/tts_response](http://localhost:8001/tts_response)

### API 端点说明

- **更换模型**: 用于切换当前 TTS 模型。
- **模型预测**: 用于请求 TTS 生成音频。

---

## 2. 大语言模型 (LLM) API

### 启动命令

要启动 LLM API 服务，有两种方式：

1. 使用 `fastapi` 命令：

    ```bash
    fastapi dev api/llm_api.py --host 0.0.0.0 --port 8002
    ```

2. 直接运行 Python 脚本：

    ```bash
    python api/llm_api.py
    ```

启动服务后，运行客户端脚本来测试 API：

```bash
python api/llm_client.py
```

### API 端点

- **基础 URL**: `http://localhost:8002`
- **API 文档**: [http://localhost:8002/docs](http://localhost:8002/docs)
- **更换模型**: [http://localhost:8002/change_model](http://localhost:8002/change_model)
- **模型预测**: [http://localhost:8002/llm_response](http://localhost:8002/llm_response)

### API 端点说明

- **更换模型**: 用于切换当前 LLM 模型。
- **模型预测**: 用于请求 LLM 生成文本。

---

## 3. 对话生成 (Talker) API

### 启动命令

要启动 Talker API 服务，有两种方式：

1. 使用 `fastapi` 命令：

    ```bash
    fastapi dev api/talker_api.py --host 0.0.0.0 --port 8003
    ```

2. 直接运行 Python 脚本：

    ```bash
    python api/talker_api.py
    ```

启动服务后，运行客户端脚本来测试 API：

```bash
python api/talker_client.py
```

生成的 mp4 音频文件会保存在 `output` 文件夹下。

### API 端点

- **基础 URL**: `http://localhost:8003`
- **API 文档**: [http://localhost:8003/docs](http://localhost:8003/docs)
- **更换模型**: [http://localhost:8003/change_model](http://localhost:8003/change_model)
- **模型预测**: [http://localhost:8003/talker_response](http://localhost:8003/talker_response)

### API 端点说明

- **更换模型**: 用于切换当前 Talker 模型。
- **模型预测**: 用于请求 Talker 生成对话。

---

## 常见问题

### 如何测试 API？

可以通过运行提供的客户端脚本（`tts_client.py`, `llm_client.py`, `talker_client.py`）来测试各个 API。确保相应的 API 服务已经在指定的端口上运行。

### 如何处理 API 错误？

请检查 API 服务的日志以获取详细的错误信息。如果出现问题，确保请求格式和内容符合 API 文档中的规范。

### API 是否需要认证？

当前示例 API 文档中未提及认证。如果有认证需求，请根据实际情况添加相应的认证机制，并更新文档说明。

### 如何查看 API 文档？

启动 API 服务后，可以在以下 URL 中查看 API 文档：

- TTS API: [http://localhost:8001/docs](http://localhost:8001/docs)
- LLM API: [http://localhost:8002/docs](http://localhost:8002/docs)
- Talker API: [http://localhost:8003/docs](http://localhost:8003/docs)

---

