# 设备运行端口 (Device running port)
port = 6006
# api运行端口及IP (API running port and IP)
mode = 'api' # api 需要先运行Linly-api-fast.py，暂时仅仅适用于Linly
ip = '127.0.0.1' 
api_port = 7871

# L模型路径 (Linly model path) 已不用了
mode = 'offline'
model_path = 'Qwen/Qwen-1_8B-Chat'

# ssl证书 (SSL certificate) 麦克风对话需要此参数
# 最好调整为绝对路径
ssl_certfile = "./https_cert/cert.pem"
ssl_keyfile = "./https_cert/key.pem"