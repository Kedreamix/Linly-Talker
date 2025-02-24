from .Whisper import WhisperASR
from .FunASR import FunASR
try:
    from .OmniSenseVoice import OmniSenseVoice
except:
    print('请先安装OmniSenseVoice, pip install -r ./ASR/requirements_OmniSenseVoice.txt')

__all__ = ['WhisperASR', 'FunASR', 'OmniSenseVoice']
