try:
    from .GPT_SoVITS import GPT_SoVITS
    print("GPT_SoVITS导入成功")
except Exception as e:
    print("GPT_SoVITS导入失败，原因：", e)
    print("使用GPT-SoVITS语音克隆前需要安装对应的环境，请执行 pip install -r VITS/requirements_vits.txt")

try:
    from .CosyVoice import CosyVoiceTTS
    print("CosyVoice导入成功")
except Exception as e:
    print("CosyVoice导入失败，原因：", e)
    print("使用CosyVoice语音克隆前需要安装对应的环境")

# try:
#     from .XTTS import XTTS
#     print("XTTS导入成功")
# except Exception as e:
#     print("XTTS导入失败，原因：", e)
#     print("使用XTTS语音克隆前需要安装对应的环境，请执行 pip install -r VITS/requirements_xtts.txt")x2xw