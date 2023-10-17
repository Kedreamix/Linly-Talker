import gradio as gr
from Asr import OpenAIASR

def main():
    asr = OpenAIASR('base')

    def transcribe_audio(audio):
        input_text = asr.transcribe(audio)
        return input_text

    audio_input = gr.Microphone()
    text_output = gr.Textbox()

    gr.Interface(fn=transcribe_audio, inputs=audio_input, outputs=text_output, title="麦克风语音转录").launch()

if __name__ == '__main__':
    main()
