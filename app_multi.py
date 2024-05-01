import os
import random 
import time
import gradio as gr
from zhconv import convert
from LLM import LLM
from ASR import WhisperASR
from TFG import SadTalker 
from TTS import EdgeTTS

from src.cost_time import calculate_time
from configs import *
os.environ["GRADIO_TEMP_DIR"]= './temp'

description = """<p style="text-align: center; font-weight: bold;">
    <span style="font-size: 28px;">Linly 智能多轮对话系统 (Linly-Talker)</span>
    <br>
    <span style="font-size: 18px;" id="paper-info">
        [<a href="https://zhuanlan.zhihu.com/p/671006998" target="_blank">知乎</a>]
        [<a href="https://www.bilibili.com/video/BV1rN4y1a76x/" target="_blank">bilibili</a>]
        [<a href="https://github.com/Kedreamix/Linly-Talker" target="_blank">GitHub</a>]
        [<a herf="https://kedreamix.github.io/" target="_blank">个人主页</a>]
    </span>
    <br> 
    <span>Linly-Talker 是一款智能 AI 对话系统，结合了大型语言模型 (LLMs) 与视觉模型，是一种新颖的人工智能交互方式。</span>
</p>
"""
# 设置默认system
default_system = '你是一个很有帮助的助手'

# 设定默认参数值，可修改
source_image = r'example.png'
blink_every = True
size_of_image = 256
preprocess_type = 'crop'
facerender = 'facevid2vid'
enhancer = False
is_still_mode = False
# pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0)
pic_path = "./inputs/girl.png"
crop_pic_path = "./inputs/first_frame_dir_girl/girl.png"
first_coeff_path = "./inputs/first_frame_dir_girl/girl.mat"
crop_info = ((403, 403), (19, 30, 502, 513), [40.05956541381802, 40.17324339233366, 443.7892505041507, 443.9029284826663])

# exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1)
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
        question = 'Gradio存在一些bug，麦克风模式有时候可能音频还未传入，请重新点击一下语音识别即可'
        gr.Warning(question)
    return question

@calculate_time
def LLM_response(question, voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0):
    answer = llm.generate(question)
    print(answer)
    try:
        tts.predict(answer, voice, rate, volume, pitch , 'answer.wav', 'answer.vtt')
    except:
        os.system(f'edge-tts --text "{answer}" --voice {voice} --write-media answer.wav')
    return 'answer.wav', 'answer.vtt', answer

@calculate_time
def Talker_response(text, voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, batch_size = 2):
    voice = 'zh-CN-XiaoxiaoNeural' if voice not in tts.SUPPORTED_VOICE else voice
    talker = SadTalker(lazy_load=True)
    driven_audio, driven_vtt, _ = LLM_response(text, voice, rate, volume, pitch)
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

def chat_response(system, message, history):
    # response = llm.generate(message)
    response, history = llm.chat(system, message, history)
    print(history)
    # 流式输出
    for i in range(len(response)):
        time.sleep(0.01)
        yield "", history[:-1] + [(message, response[:i+1])]
    return "", history

def human_respone(history, voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0, batch_size = 2):
    response = history[-1][1]
    driven_audio, video_vtt = 'answer.wav', 'answer.vtt'
    voice = 'zh-CN-XiaoxiaoNeural' if voice not in tts.SUPPORTED_VOICE else voice
    tts.predict(response, voice, rate, volume, pitch, driven_audio, video_vtt)
    pose_style = random.randint(0, 45) # 随机选择
    video_path = talker.test(pic_path,
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

    return video_path, video_vtt

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    llm.clear_history()
    return system, system, []

def clear_session():
    # clear history
    llm.clear_history()
    return '', []

def main():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(description)
        with gr.Row():   
            with gr.Column():
                with gr.Accordion("Advanced Settings(高级设置) ",
                                        open=False):
                    voice = gr.Dropdown(tts.SUPPORTED_VOICE, 
                                        value='zh-CN-XiaoxiaoNeural', 
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
                                        value=1,
                                        step=1,
                                        label='Talker Batch size')
                video = gr.Video(label = '数字人问答', scale = 0.5)
                video_button = gr.Button("🎬 生成数字人视频（对话后）", variant = 'primary')
            
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
                        system_input = gr.Textbox(value=default_system, lines=1, label='System (设定角色)')
                    with gr.Column(scale=1):
                        modify_system = gr.Button("🛠️ 设置system并清除历史对话", scale=2)
                    system_state = gr.Textbox(value=default_system, visible=False)

                chatbot = gr.Chatbot(height=400, show_copy_button=True)
                audio = gr.Audio(sources=['microphone','upload'], type="filepath", label='语音对话', autoplay=True)
                asr_text = gr.Button('🎤 语音识别（语音对话后点击）')
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")
                asr_text.click(fn=Asr,inputs=[audio],outputs=[msg])
                
                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    sumbit = gr.Button("🚀 发送", variant = 'primary')
                    
            # # 设置按钮的点击事件。当点击时，调用上面定义的 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            sumbit.click(chat_response, inputs=[system_input, msg, chatbot], 
                         outputs=[msg, chatbot])
            
            # 点击后清空后端存储的聊天记录
            clear_history.click(fn = clear_session, outputs = [msg, chatbot])
            
            # 设置system并清除历史对话
            modify_system.click(fn=modify_system_session,
                        inputs=[system_input],
                        outputs=[system_state, system_input, chatbot])
            
            video_button.click(fn = human_respone, inputs = [chatbot, voice, rate, volume, pitch, batch_size], outputs = [video])
            
        with gr.Row(variant='panel'):
            with gr.Column():
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
                    # fn = Talker_response,
                    inputs = [msg],
                    # outputs=[gen_video],
                    # cache_examples = True,
                )
    return inference


    
if __name__ == "__main__":
    # llm = LLM(mode='offline').init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf')
    # llm = LLM(mode='offline').init_model('Gemini', 'gemini-pro', api_key = "your api key")
    # llm = LLM(mode='offline').init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    llm = LLM(mode=mode).init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    talker = SadTalker(lazy_load=True)
    asr = WhisperASR('base')
    tts = EdgeTTS()
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