'''
https://github.com/openai/whisper
pip install -U openai-whisper
'''
import whisper
import sys
sys.path.append('./')
from src.cost_time import calculate_time 

class WhisperASR:
    def __init__(self, model_path):
        self.LANGUAGES = {
            "en": "english",
            "zh": "chinese",
            "de": "german",
            "es": "spanish",
            "ru": "russian",
            "ko": "korean",
            "fr": "french",
            "ja": "japanese",
            "pt": "portuguese",
            "tr": "turkish",
            "pl": "polish",
            "ca": "catalan",
            "nl": "dutch",
            "ar": "arabic",
            "sv": "swedish",
            "it": "italian",
            "id": "indonesian",
            "hi": "hindi",
            "fi": "finnish",
            "vi": "vietnamese",
            "he": "hebrew",
            "uk": "ukrainian",
            "el": "greek",
            "ms": "malay",
            "cs": "czech",
            "ro": "romanian",
            "da": "danish",
            "hu": "hungarian",
            "ta": "tamil",
            "no": "norwegian",
            "th": "thai",
            "ur": "urdu",
            "hr": "croatian",
            "bg": "bulgarian",
            "lt": "lithuanian",
            "la": "latin",
            "mi": "maori",
            "ml": "malayalam",
            "cy": "welsh",
            "sk": "slovak",
            "te": "telugu",
            "fa": "persian",
            "lv": "latvian",
            "bn": "bengali",
            "sr": "serbian",
            "az": "azerbaijani",
            "sl": "slovenian",
            "kn": "kannada",
            "et": "estonian",
            "mk": "macedonian",
            "br": "breton",
            "eu": "basque",
            "is": "icelandic",
            "hy": "armenian",
            "ne": "nepali",
            "mn": "mongolian",
            "bs": "bosnian",
            "kk": "kazakh",
            "sq": "albanian",
            "sw": "swahili",
            "gl": "galician",
            "mr": "marathi",
            "pa": "punjabi",
            "si": "sinhala",
            "km": "khmer",
            "sn": "shona",
            "yo": "yoruba",
            "so": "somali",
            "af": "afrikaans",
            "oc": "occitan",
            "ka": "georgian",
            "be": "belarusian",
            "tg": "tajik",
            "sd": "sindhi",
            "gu": "gujarati",
            "am": "amharic",
            "yi": "yiddish",
            "lo": "lao",
            "uz": "uzbek",
            "fo": "faroese",
            "ht": "haitian creole",
            "ps": "pashto",
            "tk": "turkmen",
            "nn": "nynorsk",
            "mt": "maltese",
            "sa": "sanskrit",
            "lb": "luxembourgish",
            "my": "myanmar",
            "bo": "tibetan",
            "tl": "tagalog",
            "mg": "malagasy",
            "as": "assamese",
            "tt": "tatar",
            "haw": "hawaiian",
            "ln": "lingala",
            "ha": "hausa",
            "ba": "bashkir",
            "jw": "javanese",
            "su": "sundanese",
        }
        self.model = whisper.load_model(model_path)

    @calculate_time
    def transcribe(self, audio_file):
        result = self.model.transcribe(audio_file)
        return result["text"]


if __name__ == "__main__":
    import os
    # 创建ASR对象并进行语音识别
    model_path = "./Whisper/tiny.pt"  # 模型路径
    audio_file = "output.wav"  # 音频文件路径
    if not os.path.exists(audio_file):
        os.system('edge-tts --text "hello" --write-media output.wav')
    asr = WhisperASR(model_path)
    print(asr.transcribe(audio_file))