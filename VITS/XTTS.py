
import os
import subprocess
import uuid
import time
import torch
import torchaudio

# langid is used to detect language for longer text
# Most users expect text to be their own language, there is checkbox to disable it
import langid
import re
import gradio as gr
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir

class XTTS():
    def __init__(self):
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        # ModelManager().download_model(model_name)
        model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
        # print("XTTS downloaded")

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))

        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_path=os.path.join(model_path, "model.pth"),
            vocab_path=os.path.join(model_path, "vocab.json"),
            eval=True,
            use_deepspeed=True,
        )
        self.model.cuda()

        self.supported_languages = config.languages

    def predict(self, 
        prompt,
        language,
        audio_file_pth,
        voice_cleanup,
    ):
        # 模型不支持语言
        if language not in self.supported_languages:
            gr.Warning(
                f"Language you put {language} in is not in is not in our Supported Languages, please choose from dropdown"
            )

            return (
                None,
                None,
                None,
                None,
            )

        language_predicted = langid.classify(prompt)[
            0
        ].strip()  # strip need as there is space at end!

        # tts expects chinese as zh-cn
        if language_predicted == "zh":
            # we use zh-cn
            language_predicted = "zh-cn"

        print(f"Detected language:{language_predicted}, Chosen language:{language}")

        speaker_wav = audio_file_pth

        # Filtering for microphone input, as it has BG noise, maybe silence in beginning and end
        # This is fast filtering not perfect

        # Apply all on demand
        lowpassfilter = denoise = trim = loudness = True

        if lowpassfilter:
            lowpass_highpass = "lowpass=8000,highpass=75,"
        else:
            lowpass_highpass = ""

        if trim:
            # better to remove silence in beginning and end for microphone
            trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
        else:
            trim_silence = ""

        if voice_cleanup:
            try:
                out_filename = (
                    speaker_wav + str(uuid.uuid4()) + ".wav"
                )  # ffmpeg to know output format

                # we will use newer ffmpeg as that has afftn denoise filter
                shell_command = f"ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}".split(
                    " "
                )

                command_result = subprocess.run(
                    [item for item in shell_command],
                    capture_output=False,
                    text=True,
                    check=True,
                )
                speaker_wav = out_filename
                print("Filtered microphone input")
            except subprocess.CalledProcessError:
                # There was an error - command exited with non-zero code
                print("Error: failed filtering, use original microphone input")
        else:
            speaker_wav = speaker_wav

        if len(prompt) < 2:
            gr.Warning("Please give a longer prompt text")
            return (
                None,
                None,
                None,
                None,
            )
        
        metrics_text = ""
        t_latent = time.time()

        # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
        try:
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = self.model.get_conditioning_latents(audio_path=speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
        except Exception as e:
            print("Speaker encoding error", str(e))
            gr.Warning(
                "It appears something wrong with reference, did you unmute your microphone?"
            )
            return (
                None,
                None,
                None,
            )

        latent_calculation_time = time.time() - t_latent
        # metrics_text=f"Embedding calculation time: {latent_calculation_time:.2f} seconds\n"

        # temporary comma fix
        prompt= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2",prompt)

        wav_chunks = []
        ## Direct mode
        
        print("I: Generating new audio...")
        t0 = time.time()
        out = self.model.inference(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
        )
        inference_time = time.time() - t0
        print(f"I: Time to generate audio: {round(inference_time*1000)} milliseconds")
        metrics_text+=f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
        real_time_factor= (time.time() - t0) / out['wav'].shape[-1] * 24000
        print(f"Real-time factor (RTF): {real_time_factor}")
        metrics_text+=f"Real-time factor (RTF): {real_time_factor:.2f}\n"
        torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
        return (
            "output.wav",
            metrics_text,
            speaker_wav,
        )

title = "语音克隆 Coqui🐸 XTTS"

examples = [
    [
        "Once when I was six years old I saw a magnificent picture",
        "en",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image",
        "fr",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Als ich sechs war, sah ich einmal ein wunderbares Bild",
        "de",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Cuando tenía seis años, vi una vez una imagen magnífica",
        "es",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Quando eu tinha seis anos eu vi, uma vez, uma imagem magnífica",
        "pt",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Kiedy miałem sześć lat, zobaczyłem pewnego razu wspaniały obrazek",
        "pl",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno",
        "it",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Bir zamanlar, altı yaşındayken, muhteşem bir resim gördüm",
        "tr",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Когда мне было шесть лет, я увидел однажды удивительную картинку",
        "ru",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Toen ik een jaar of zes was, zag ik op een keer een prachtige plaat",
        "nl",
        "examples/male.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "Když mi bylo šest let, viděl jsem jednou nádherný obrázek",
        "cs",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "当我还只有六岁的时候， 看到了一副精彩的插画",
        "zh-cn",
        "examples/female.wav",
        None,
        False,
        False,
        False,
        True,
    ],
    [
        "かつて 六歳のとき、素晴らしい絵を見ました",
        "ja",
        "examples/female.wav",
        None,
        False,
        True,
        False,
        True,
    ],
    [
        "한번은 내가 여섯 살이었을 때 멋진 그림을 보았습니다.",
        "ko",
        "examples/female.wav",
        None,
        False,
        True,
        False,
        True,
    ],
        [
        "Egyszer hat éves koromban láttam egy csodálatos képet",
        "hu",
        "examples/male.wav",
        None,
        False,
        True,
        False,
        True,
    ],
]



def main():
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    ## 语音克隆 (Coqui TTS)
                    """
                )
            with gr.Column():
                # 用于对齐图像的占位符
                pass

        with gr.Row():
            with gr.Column():
                gr.Markdown("文本提示: 一次一两个句子最好。最多200个文本字符。")
            with gr.Column():
                gr.Markdown("语言: 选择用于合成语音的输出语言。")

        with gr.Row():
            with gr.Column():
                input_text_gr = gr.Textbox(
                    label="文本提示",
                    info="一次一两个句子最好。最多200个文本字符。",
                    value="嗨，我是你的新语音克隆。请尽量上传高质量的音频。",
                )
                language_gr = gr.Dropdown(
                    label="语言",
                    info="选择用于合成语音的输出语言",
                    choices=[
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "pl",
                        "tr",
                        "ru",
                        "nl",
                        "cs",
                        "ar",
                        "zh-cn",
                        "ja",
                        "ko",
                        "hu",
                        "hi"
                    ],
                    max_choices=1,
                    value="en",
                )
                ref_gr = gr.Audio(
                    label="参考音频",
                    type="filepath",
                    value="examples/female.wav",
                    sources=["microphone", "upload"],
                )
                clean_ref_gr = gr.Checkbox(
                    label="清理参考语音",
                    value=False,
                    info="如果您的麦克风或参考语音有噪音，此选项可以改善输出。",
                )
                tts_button = gr.Button("发送", elem_id="send-btn", visible=True)


            with gr.Column():
                audio_gr = gr.Audio(label="合成音频", autoplay=True)
                out_text_gr = gr.Text(label="指标")
                ref_audio_gr = gr.Audio(label="使用的参考音频")
        with gr.Row():
            gr.Examples(examples,
                        label="示例",
                        inputs=[input_text_gr, language_gr, ref_gr, clean_ref_gr],
                        outputs=[audio_gr, out_text_gr, ref_audio_gr],
                        fn=XTTS().predict,
                        cache_examples=False,)
        tts_button.click(XTTS().predict, [input_text_gr, language_gr, ref_gr, clean_ref_gr], outputs=[audio_gr, out_text_gr, ref_audio_gr])
    return demo

if __name__ == "__main__":  
    demo = main()  
    demo.launch()