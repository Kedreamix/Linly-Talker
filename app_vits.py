import os
import random 
import gradio as gr
from zhconv import convert
from LLM import LLM
from ASR import WhisperASR, FunASR
from TFG import SadTalker 
from TTS import EdgeTTS
from VITS import GPT_SoVITS
from src.cost_time import calculate_time

from configs import *
os.environ["GRADIO_TEMP_DIR"]= './temp'

description = """<p style="text-align: center; font-weight: bold;">
    <span style="font-size: 28px;">Linly 智能对话系统 (Linly-Talker + GPT-SoVITS)</span>
    <br>
    <span style="font-size: 18px;" id="paper-info">
        [<a href="https://zhuanlan.zhihu.com/p/671006998" target="_blank">知乎</a>]
        [<a href="https://www.bilibili.com/video/BV1rN4y1a76x/" target="_blank">bilibili</a>]
        [<a href="https://github.com/Kedreamix/Linly-Talker" target="_blank">GitHub</a>]
        [<a herf="https://kedreamix.github.io/" target="_blank">个人主页</a>]
        [<a herf="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">GPT-SoVITS</a>]
    </span>
    <br> 
    <span>Linly-Talker 是一款智能 AI 对话系统，结合了大型语言模型 (LLMs) 与视觉模型，是一种新颖的人工智能交互方式。</span>
</p>
"""

# 设定默认参数值，可修改
source_image = r'./inputs/boy.png'
blink_every = True
size_of_image = 256
preprocess_type = 'crop'
facerender = 'facevid2vid'
enhancer = False
is_still_mode = False
pic_path = "./inputs/boy.png"
crop_pic_path = "./inputs/first_frame_dir_boy/boy.png"
first_coeff_path = "./inputs/first_frame_dir_boy/boy.mat"
crop_info = ((876, 747), (0, 0, 886, 838), [10.382158280494476, 0, 886, 747.7078990925525])

exp_weight = 1

use_ref_video = False
ref_video = None
ref_info = 'pose'
use_idle_mode = False
length_of_audio = 5

@calculate_time
def Asr(audio):
    try:
        question = asr.transcribe(audio)
        question = convert(question, 'zh-cn')
    except Exception as e:
        print("ASR Error: ", e)
        question = 'Gradio 的麦克风有时候可能音频还未传入，请重试一下'
    return question

@calculate_time
def LLM_response(question_audio, question, voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0):
    answer = llm.generate(question)
    print(answer)
    if voice in tts.SUPPORTED_VOICE:
        try:
            tts.predict(answer, voice, rate, volume, pitch , 'answer.wav', 'answer.vtt')
        except:
            os.system(f'edge-tts --text "{answer}" --voice {voice} --write-media answer.wav --write-subtitles answer.vtt')
    elif voice == "克隆烟嗓音":
        gpt_path = "../GPT-SoVITS/GPT_weights/yansang-e15.ckpt"
        sovits_path = "../GPT-SoVITS/SoVITS_weights/yansang_e16_s144.pth"
        vits.load_model(gpt_path, sovits_path)
        vits.predict(ref_wav_path = "examples/slicer_opt/vocal_output.wav_10.wav_0000846400_0000957760.wav",
                    prompt_text = "你为什么要一次一次的伤我的心啊？",
                    prompt_language = "中文",
                    text = answer,
                    text_language = "中英混合",
                    how_to_cut = "按标点符号切",
                    save_path = 'answer.wav')
    elif voice == "克隆声音":
        if question_audio is None:
            print("无声音输入，无法克隆声音")
            return None, None, None
        gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        vits.load_model(gpt_path, sovits_path)
        vits.predict(ref_wav_path = question_audio, 
                    prompt_text = question,
                    prompt_language = "中文",
                    text = answer,
                    text_language = "中英混合",
                    how_to_cut = "凑四句一切",
                    save_path = 'answer.wav')
    return 'answer.wav', None, answer

@calculate_time
def Talker_response(question_audio, text, voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, batch_size = 2):
    # voice = 'zh-CN-XiaoxiaoNeural' if voice not in tts.SUPPORTED_VOICE else voice
    # print(voice , rate , volume , pitch)
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, voice, rate, volume, pitch)
    pose_style = random.randint(0, 45)
    video = talker.test(pic_path,
                        crop_pic_path,
                        first_coeff_path,
                        crop_info,
                        source_image,
                        driven_audio,
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        use_ref_video,
                        ref_video,
                        ref_info,
                        use_idle_mode,
                        length_of_audio,
                        blink_every,
                        fps=20)
    if driven_vtt:
        return video, driven_vtt
    else:
        return video
    
def main():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(description)
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="question_audio"):
                    with gr.TabItem('对话'):
                        with gr.Column(variant='panel'):
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)

                            with gr.Accordion("Advanced Settings(高级设置语音参数) ",
                                        open=False):
                                gr.Markdown("若进行克隆声音，声音需要大于3s，小于10s，语音识别后可点击语音对话，否则无法克隆声音")
                                voice = gr.Dropdown(["克隆声音", "克隆烟嗓音"] + tts.SUPPORTED_VOICE, 
                                                    value='克隆声音', 
                                                    label="Voice")
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
                                batch_size = gr.Slider(minimum=1,
                                                    maximum=10,
                                                    value=2,
                                                    step=1,
                                                    label='Talker Batch size')
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                            asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
                            
                        # with gr.Column(variant='panel'):
                        #     input_text = gr.Textbox(label="Input Text", lines=3)
                        #     text_button = gr.Button("文字对话", variant='primary')
                        
                
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.TabItem('数字人问答'):
                        gen_video = gr.Video(label="Generated video", format="mp4", scale=1, autoplay=True)
                video_button = gr.Button("提交", variant='primary')
            video_button.click(fn=Talker_response,inputs=[question_audio, input_text,voice, rate, volume, pitch, batch_size],outputs=[gen_video])

        with gr.Row():
            with gr.Column(variant='panel'):
                    gr.Markdown("## Text Examples")
                    examples =  ['应对压力最有效的方法是什么？',
                        '如何进行时间管理？',
                        '为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？',
                        '近日，苹果公司起诉高通公司，状告其未按照相关合约进行合作，高通方面尚未回应。这句话中“其”指的是谁？',
                        '三年级同学种树80颗，四、五年级种的棵树比三年级种的2倍多14棵，三个年级共种树多少棵?',
                        '撰写一篇交响乐音乐会评论，讨论乐团的表演和观众的整体体验。',
                        '翻译成中文：Luck is a dividend of sweat. The more you sweat, the luckier you get.',
                        ]
                    gr.Examples(
                        examples = examples,
                        fn = Talker_response,
                        inputs = [input_text],
                        outputs=[gen_video],
                        # cache_examples = True,
                    )
    return inference


    
if __name__ == "__main__":
    # llm = LLM(mode='offline').init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf')
    # llm = LLM(mode='offline').init_model('Gemini', 'gemini-pro', api_key = "your api key")
    # llm = LLM(mode='offline').init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    llm = LLM(mode=mode).init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    talker = SadTalker(lazy_load=True)
    # asr = WhisperASR('base')
    asr = FunASR()
    tts = EdgeTTS()
    vits = GPT_SoVITS()
    gr.close_all()
    demo = main()
    demo.queue()
    # demo.launch()
    demo.launch(server_name=ip, # 本地端口localhost:127.0.0.1 全局端口转发:"0.0.0.0"
                server_port=port,
                # 似乎在Gradio4.0以上版本可以不使用证书也可以进行麦克风对话
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
                ssl_verify=False,                
                debug=True)