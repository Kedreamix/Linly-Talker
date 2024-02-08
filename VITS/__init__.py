try:
    from .GPT_SoVITS import GPT_SoVITS
except:
    print("使用GPT-SoVITS语音克隆前需要安装对应的环境，请执行 pip install -r VITS/requirements_vits.txt")
    
try:
    from .XTTS import XTTS
except:
    print("使用XTTS语音克隆前需要安装对应的环境，请执行 pip install -r VITS/requirements_xtts.txt")