from .SadTalker import SadTalker
from .Wav2Lip import Wav2Lip
from .Wav2Lipv2 import Wav2Lipv2
try:
    from .NeRFTalk import NeRFTalk
except Exception as e:
    print("NeRFTalk导入失败，原因：", e)
    print("使用NeRFTalk前需要安装对应的环境")
try:
    from .MuseTalk import MuseTalk_RealTime
except Exception as e:
    print("MuseTalk导入失败，原因：", e)
    print("使用MuseTalk前需要安装对应的环境")