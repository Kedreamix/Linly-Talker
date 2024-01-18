# 设备运行端口 (Device running port)
port = 7870
# api运行端口 (API running port)
api_port = 7871
# Linly模型路径 (Linly model path)
# mode = 'api' # api 需要先运行Linly-api-fast.py
mode = 'offline'
model_path = 'Linly-AI/Chinese-LLaMA-2-7B-hf'
# ssl证书 (SSL certificate) 麦克风对话需要此参数
ssl_certfile = "/path/to/Linly-Talker/https_cert/cert.pem"
ssl_keyfile = "/path/to/Linly-Talker/https_cert/key.pem"