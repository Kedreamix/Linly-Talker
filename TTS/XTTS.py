import torch
import tempfile
from TTS.api import TTS


class XTTSTalker():
    def __init__(self) -> None:
        model_list = TTS().list_models()
        print(model_list)
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1.1").to(device)

    def test(self, text, language='en'):

        tempf  = tempfile.NamedTemporaryFile(
                delete = False,
                suffix = ('.'+'wav'),
            )
        # wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
        self.tts.tts_to_file(text, language=language, file_path="./speaker.wav")
        
        return tempf.name
    
if __name__ == "__main__":
    tts = XTTSTalker()
    print(tts.test("Hello world!"))