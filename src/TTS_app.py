# import gradio as gr
# from TTS import EdgeTTS
# import librosa
# import os

# title = "文本转语音"

# def generateAudio(text):
#     tts = EdgeTTS()
#     audio_file = tts.predict(text, 'zh-CN-XiaoxiaoNeural', '+0%', '+0%', "output.wav")
#     audio, sr = librosa.load(path=audio_file)
#     return sr,audio

# app = gr.Interface(
#     fn=generateAudio, 
#     inputs="text", 
#     outputs="audio", 
#     title=title,
#     # examples=[os.path.join(os.path.dirname(__file__),"output.wav")]
#     )
# if __name__ == "__main__":
#     app.launch()

import gradio as gr
import librosa
# from TTS import EdgeTTS
import os
def generateAudio(text):
    # tts = EdgeTTS()
    # VOICE = "zh-CN-XiaoxiaoNeural"
    # OUTPUT_FILE = "tts.wav"
    # audio_file = tts.predict(text, VOICE, '+0%', '+0%', OUTPUT_FILE)
    # edge-tts --text "Hello, world!" --write-media hello.mp3 --write-subtitles hello.vtt
    os.system(f'proxychains4 edge-tts --text "{text}" --voice zh-CN-XiaoxiaoNeural --write-media tts.wav')
    audio, sr = librosa.load(path='tts.wav')
    return sr,audio


demo = gr.Interface(
    fn=generateAudio, 
    inputs="text", 
    outputs='audio',
    )
    
if __name__ == "__main__":
    demo.launch()  