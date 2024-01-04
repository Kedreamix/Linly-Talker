import os
import gradio as gr
from zhconv import convert
from src.LLM import *
from src.Asr import OpenAIASR
from src.gradio_demo import SadTalker 
import time
import random 
description = """<p style="text-align: center; font-weight: bold;">
        <span style="font-size: 28px">Linly Talker</span>
        <br>
        <span style="font-size: 18px" id="paper-info">
        Linly-Talker is an intelligent AI system that combines large language models (LLMs) with visual models to create a novel human-AI interaction method. 
        <br> 
    </p>"""


VOICES = ['zh-CN-XiaoxiaoNeural', 
        'zh-CN-XiaoyiNeural', 
        'zh-CN-YunjianNeural', 
        'zh-CN-YunxiNeural', 
        'zh-CN-YunxiaNeural', 
        'zh-CN-YunyangNeural', 
        'zh-HK-HiuGaaiNeural', 
        'zh-HK-HiuMaanNeural', 
        'zh-HK-WanLungNeural', 
        'zh-TW-HsiaoChenNeural',  
        'zh-TW-YunJheNeural', 
        'zh-TW-HsiaoYuNeural',
        'en-US-AnaNeural', 
        'en-US-AriaNeural', 
        'en-US-ChristopherNeural', 
        'en-US-EricNeural', 
        'en-US-GuyNeural', 
        'en-US-JennyNeural', 
        'en-US-MichelleNeural',
        ]

source_image = r'example.png'
batch_size = 8
blink_every = True
size_of_image = 256
preprocess_type = 'crop'
facerender = 'facevid2vid'
enhancer = False
is_still_mode = False
# pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0)

# exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1)
exp_weight = 1

use_ref_video = False
ref_video = None
ref_info = 'pose'
use_idle_mode = False
length_of_audio = 5
voice = 'zh-CN-XiaoxiaoNeural'


def asr(audio):
    #sr, data = audio
    #audio = "audio_output.wav"
    #sf.write(audio, data, samplerate=sr)  # 以44100Hz的采样率保存为WAV文件
    #print(audio)
    question = openaiasr.transcribe(audio)
    question = convert(question, 'zh-cn')
    return question
# def asr(audio):
#     # openaiasr = OpenAIASR('base')
#     # question = openaiasr.transcribe(audio)
#     # question = convert(question, 'zh-cn')
#     question = funasr.inference(audio)
#     return question

 
def llm_response(question):
    voice = 'zh-CN-XiaoxiaoNeural'
    #answer = llm.predict(question)
    answer = llm.generate(question)
    print(answer)
    # 默认保存为answer.wav
    os.system(f'edge-tts --text "{answer}" --voice {voice} --write-media answer.wav')
    #audio, sr = librosa.load(path='answer.wav')
    return 'results/answer.wav', answer


def asr_response(audio):
    
    question = asr(audio)
    llm_response(question)
    pose_style = random.randint(0, 45)
    video = sad_talker.test(source_image,
                        'answer.wav',
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
                        blink_every)
    return video

def text_response(text):
    s = time.time()
    sad_talker = SadTalker(lazy_load=True)
    llm_response(text)
    e = time.time()
    print("Using Time", e-s)
    pose_style = random.randint(0, 45)
    s = time.time()
    video = sad_talker.test(source_image,
                        'answer.wav',
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
                        blink_every)
    e = time.time()
    print("Using Time", e-s)
    # print(video)
    return video

def main():
    
    with gr.Blocks(analytics_enabled=False) as inference:
        gr.HTML(description)
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="question_audio"):
                    with gr.TabItem('对话'):
                        with gr.Column(variant='panel'):
                            question_audio = gr.Audio(source="microphone",type="filepath")
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            asr_text = gr.Button('语音识别')
                            asr_text.click(fn=asr,inputs=[question_audio],outputs=[input_text])
                            
                        # with gr.Column(variant='panel'):
                        #     input_text = gr.Textbox(label="Input Text", lines=3)
                        #     text_button = gr.Button("文字对话", variant='primary')
                        
                
            with gr.Column(variant='panel'): 
        
                with gr.Tabs(elem_id="sadtalker_genearted"):
                    with gr.TabItem('数字人问答'):
                        gen_video = gr.Video(label="Generated video", format="mp4", scale=1)
                    video_button = gr.Button("提交",variant='primary')
            video_button.click(fn=text_response,inputs=[input_text],outputs=[gen_video])
            
            # text_button.click(fn=text_response,inputs=[input_text],outputs=[gen_video])
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
                        inputs = [input_text],
                        fn = text_response,
                        outputs=[gen_video]
                    )
    return inference


    
if __name__ == "__main__":
    # funasr = FunASR()
    # local 
    # llm = Linly(mode='offline',model_path="./Chinese-LLaMA-2-7B-hf/")
    # api

    # llm = Gemini(model_path='gemini-pro', api_key=None, proxy_url=None) # 需要自己加入google的apikey
    # llm = Qwen(mode='offline',model_path="Qwen/Qwen-1_8B-Chat")
    # 自动下载
    # llm = Linly(mode='offline',model_path="Linly-AI/Chinese-LLaMA-2-7B-hf")
    # 手动下载指定路径
    llm = Linly(mode='offline',model_path="./Chinese-LLaMA-2-7B-hf")
    sad_talker = SadTalker(lazy_load=True)
    openaiasr = OpenAIASR('base')
    gr.close_all()
    demo = main()
    demo.queue()
    # demo.launch()
    demo.launch(server_name="0.0.0.0", 
                server_port=7870, 
                ssl_certfile="/path/to/cert.pem", 
                ssl_keyfile="/path/to/key.pem",
                ssl_verify=False)
