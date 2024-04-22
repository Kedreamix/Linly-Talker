from .EdgeTTS import EdgeTTS

try:
    from .PaddleTTS import PaddleTTS
except Exception as e:
    print("PaddleTTS Error: ", e)
    print("如果使用PaddleTTS，请先安装PaddleTTS环境")
    print("pip install -r requirements_paddle.txt")