import librosa
import gradio as gr
from PaddleTTS import PaddleTTS
import os
os.environ["GRADIO_TEMP_DIR"]= './temp'
title = "PaddleTTS WebUI"
tts = PaddleTTS()

def generateAudio(text, am = 'fastspeech2', voc = 'PWGan', lang = 'zh', male=False, spk_id = 174):
    print("text:", text, "am:", am, "voc:", voc, "lang:", lang, "male:", male, "spk_id:", spk_id)
    audio_file = tts.predict(text, am, voc, spk_id, lang, male)
    audio, sr = librosa.load(path=audio_file)
    return gr.make_waveform(
                audio=audio_file,
            ),(sr, audio)


def main():
    with gr.Blocks(title=title) as demo:
        with gr.Row():
            gr.HTML("<center><h1>PaddleTTS WebUI</h1></center>")
        with gr.Row():
            with gr.Column():
                text = gr.Text(label = "Text to be spoken")
                am = gr.Dropdown(["FastSpeech2"], label="声学模型选择", value = 'FastSpeech2')
                voc = gr.Dropdown(["PWGan", "HifiGan"], label="声码器选择", value = 'PWGan')
                lang = gr.Dropdown(["zh", "en", "mix", "canton"], label="语言选择", value = 'zh')
                male = gr.Checkbox(label="男声(Male)", value=False)
            
            with gr.Column():
                video = gr.Video(label="Waveform Visual")
                audio = gr.Audio(label = "Audio file")
            
        generate = gr.Button("Generate Audio", variant="primary")
        generate.click(generateAudio, 
                        inputs=[text, am, voc, lang, male], 
                        outputs=[video, audio],
                        ) 
        gr.Markdown("## Text Examples")
        gr.Examples(
            examples=[
               ["Hello World", "FastSpeech2", "PWGan", "en", False],
               ["Hello World", "FastSpeech2", "PWGan", "en", True],
               ["你好世界", "FastSpeech2", "PWGan", "zh", True],
               ["你好世界", "FastSpeech2", "PWGan", "zh", False],
               ["你好世界Hello World", "FastSpeech2", "PWGan", "mix", False],
               ["你好世界", "FastSpeech2", "PWGan", "canton", False],
            ],
            fn=generateAudio,
            inputs=[text, am, voc, lang, male],
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