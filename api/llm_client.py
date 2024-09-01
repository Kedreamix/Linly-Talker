import os
import requests

llm_service_host = os.environ.get("LLM_SERVICE_HOST", "localhost")
llm_service_port = os.environ.get("LLM_SERVICE_PORT", 8002)

# API endpoint URLs
CHANGE_MODEL_URL = f"http://{llm_service_host}:{llm_service_port}/llm_change_model/"
LLM_RESPONSE_URL = f"http://{llm_service_host}:{llm_service_port}/llm_response/"

def change_model(model_name, gemini_apikey='', openai_apikey='', proxy_url=None):
    """请求更换LLM模型"""
    params = {
        "model_name": model_name,
        "gemini_apikey": gemini_apikey,
        "openai_apikey": openai_apikey,
        "proxy_url": proxy_url,
    }
    response = requests.post(CHANGE_MODEL_URL, params=params)
    if response.status_code == 200:
        print(f"模型更换成功: {response.json()}")
    else:
        print(f"模型更换失败: {response.status_code}, {response.text}")

def request_llm_response(payload):
    """请求LLM生成回答"""
    response = requests.post(LLM_RESPONSE_URL, json=payload)
    if response.status_code == 200:
        print(f"LLM 回复成功: {response.json()}")
    else:
        print(f"LLM 回复失败: {response.status_code}, {response.text}")

if __name__ == "__main__":
    # 要测试的模型列表
    models = [
        # "GPT4Free",
        "Qwen",
    ]

    # 循环更换模型并生成LLM回复
    for model_name in models:
        print(f"切换到模型: {model_name}")
        change_model(model_name, openai_apikey="your_openai_api_key")

        # 请求LLM生成回答
        payload = {
            "question": "请问什么是FastAPI？",
            "model_name": model_name,
            "gemini_apikey": "",
            "openai_apikey": "your_openai_api_key",
            # "proxy_url": None
        }
        request_llm_response(payload)
        print("\n" + "-"*50 + "\n")
