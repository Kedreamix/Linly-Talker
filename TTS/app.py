import librosa
import gradio as gr
from EdgeTTS import EdgeTTS
import os
title = "TTS WebUI"
tts = EdgeTTS()

def generateAudio(text, voice, rate, volume, pitch):
    audio_file, sub_file = tts.predict(text, voice, rate, volume, pitch, "output.wav", "output.srt")
    print(text, audio_file, sub_file)
    audio, sr = librosa.load(path=audio_file)
    return gr.make_waveform(
                audio=audio_file,
            ),(sr, audio)


def main():
    with gr.Blocks(title=title) as demo:
        with gr.Row():
            gr.HTML("<center><h1>TTS WebUI</h1></center>")
        with gr.Row():
            with gr.Column():
                text = gr.Text(label = "Text to be spoken")
                voice = gr.Dropdown(tts.SUPPORTED_VOICE, label="Voice to be used", value = 'zh-CN-XiaoxiaoNeural')
                with gr.Accordion("Advanced Settings",
                                        open=True,
                                        visible=True) as parameter_article:
                    rate = gr.Slider(minimum=-100,
                                        maximum=100,
                                        value=0,
                                        step=1.0,
                                        label='Rate')
                    volume = gr.Slider(minimum=0,
                                            maximum=100,
                                            value=100,
                                            step=1,
                                            label='Volume')
                    pitch = gr.Slider(minimum=-100,
                                        maximum=100,
                                        value=0,
                                        step=1,
                                        label='Pitch')

            with gr.Column():
                video = gr.Video(label="Waveform Visual")
                audio = gr.Audio(label = "Audio file")
            
        generate = gr.Button("Generate Audio", variant="primary")
        generate.click(generateAudio, 
                        inputs=[text, voice, rate, volume, pitch], 
                        outputs=[video, audio],
                        ) 
        gr.Markdown("## Text Examples")
        gr.Examples(
            examples=[
                ['大家好，很高兴认识你们！','zh-CN-XiaoxiaoNeural'],
                ['みなさん、こんにちは！お会いできて嬉しいです！','ja-JP-NanamiNeural'],
                ['hello, Nice to meet you!','en-US-RogerNeural']
            ],
            fn=generateAudio,
            inputs=[text, voice],
            outputs=[video, audio],
        )
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7860)
    opt = parser.parse_args()
    demo = main()
    demo.launch(server_port=opt.server_port)