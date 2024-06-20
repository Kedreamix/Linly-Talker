import os 
import gradio as gr
from zhconv import convert
from src.cost_time import calculate_time
from configs import *
description = """<p style="text-align: center; font-weight: bold;">
    <span style="font-size: 28px;">Linly 智能对话系统 (Linly-Talker)</span>
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
        os.system(f'edge-tts --text "{answer}" --voice {voice} --write-media answer.wav --write-subtitles answer.vtt')
    except:
        tts.predict(answer, voice, rate, volume, pitch , 'answer.wav', 'answer.vtt')
    return 'answer.wav', 'answer.vtt', answer

@calculate_time
def Talker_response(text, voice, rate, volume, pitch, source_video, bbox_shift):
    voice = 'zh-CN-XiaoxiaoNeural' if voice not in tts.SUPPORTED_VOICE else voice
    driven_audio, driven_vtt, _ = LLM_response(text, voice, rate, volume, pitch)
    
    video = musetalker.inference_noprepare(driven_audio, 
                                            source_video, 
                                            bbox_shift) 
    
    if driven_vtt:
        return (video, driven_vtt)
    else:
        return video

def main():
    
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(description)
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                        with gr.TabItem('MuseV Video'):
                            gr.Markdown("MuseV: need help? please visit MuseVDemo to generate Video https://huggingface.co/spaces/AnchorFake/MuseVDemo")
                            with gr.Row():
                                source_video = gr.Video(label="Reference Video",sources=['upload'])
                            gr.Markdown("BBox_shift 推荐值下限，在生成初始结果后生成相应的 bbox 范围。如果结果不理想，可以根据该参考值进行调整。\n一般来说，在我们的实验观察中，我们发现正值（向下半部分移动）通常会增加嘴巴的张开度，而负值（向上半部分移动）通常会减少嘴巴的张开度。然而，需要注意的是，这并不是绝对的规则，用户可能需要根据他们的具体需求和期望效果来调整该参数。")
                            with gr.Row():
                                bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
                                bbox_shift_scale = gr.Textbox(label="bbox_shift_scale", 
                                                              value="",interactive=False)

                source_video.change(fn=musetalker.prepare_material, inputs=[source_video, bbox_shift], outputs=[source_video, bbox_shift_scale])
                with gr.Tabs(elem_id="question_audio"):
                    with gr.TabItem('对话'):
                        with gr.Column(variant='panel'):
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3, info = '文字对话')
                            with gr.Accordion("Advanced Settings",
                                        open=False,
                                        visible=True) as parameter_article:
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

                            asr_text = gr.Button('语音识别（语音对话后点击）')
                            asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
                
                            
                with gr.Tabs(): 
                    gr.Markdown("## Text Examples")
                    examples =  [
                        ['应对压力最有效的方法是什么？'],
                        ['如何进行时间管理？'],
                        ['为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？'],
                        ['近日，苹果公司起诉高通公司，状告其未按照相关合约进行合作，高通方面尚未回应。这句话中“其”指的是谁？'],
                        ['三年级同学种树80颗，四、五年级种的棵树比三年级种的2倍多14棵，三个年级共种树多少棵?'],
                        ['撰写一篇交响乐音乐会评论，讨论乐团的表演和观众的整体体验。'],
                        ['翻译成中文：Luck is a dividend of sweat. The more you sweat, the luckier you get.'],
                    ]
                    gr.Examples(
                        examples = examples,
                        inputs = [input_text],
                    )
                    
            # driven_audio = 'answer.wav'           
            with gr.Column(variant='panel'):                     
                with gr.TabItem("MuseTalk Video"):
                    gen_video = gr.Video(label="Generated video", format="mp4")
                submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                examples = [os.path.join('Musetalk/data/video', video) for video in os.listdir("Musetalk/data/video")]
                # ['Musetalk/data/video/yongen_musev.mp4', 'Musetalk/data/video/musk_musev.mp4', 'Musetalk/data/video/monalisa_musev.mp4', 'Musetalk/data/video/sun_musev.mp4', 'Musetalk/data/video/seaside4_musev.mp4', 'Musetalk/data/video/sit_musev.mp4', 'Musetalk/data/video/man_musev.mp4']
                
                gr.Markdown("## MuseV Video Examples")
                gr.Examples(
                    examples=[
                        ['Musetalk/data/video/yongen_musev.mp4', 5],
                        ['Musetalk/data/video/musk_musev.mp4', 5],
                        ['Musetalk/data/video/monalisa_musev.mp4', 5],
                        ['Musetalk/data/video/sun_musev.mp4', 5],
                        ['Musetalk/data/video/seaside4_musev.mp4', 5],
                        ['Musetalk/data/video/sit_musev.mp4', 5],
                        ['Musetalk/data/video/man_musev.mp4', 5]
                        ],
                    inputs =[source_video, bbox_shift], 
                )

            submit.click(
                fn=Talker_response,
                inputs=[input_text,
                        voice, rate, volume, pitch,
                        source_video, bbox_shift], 
                outputs=[gen_video]
                )
    return inference

def success_print(text):
    print(f"\033[1;31;42m{text}\033[0m")

def error_print(text):
    print(f"\033[1;37;41m{text}\033[0m")
    
if __name__ == "__main__":
    # llm = LLM(mode='offline').init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf')
    # llm = LLM(mode='offline').init_model('Gemini', 'gemini-pro', api_key = "your api key")
    # llm = LLM(mode='offline').init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    try:
        from LLM import LLM
        llm = LLM(mode=mode).init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    except Exception as e:
        error_print(f"LLM is not ready, error: {e}")
        error_print("如果使用LLM，请先下载有关的LLM模型")
        
    try:
        from TTS import EdgeTTS
        tts = EdgeTTS()
    except Exception as e:
        error_print(f"EdgeTTS Error: {e}")
        error_print("如果使用EdgeTTS，请先下载EdgeTTS库，测试EdgeTTS是否可用")
    
    try:
        from ASR import WhisperASR
        asr = WhisperASR('base')
    except Exception as e:
        error_print(f"ASR Error: {e}")
        error_print("如果使用ASR，请先下载ASR相关模型，如Whisper")

    try:
        from TFG import MuseTalk_RealTime
        musetalker = MuseTalk_RealTime()
        musetalker.init_model()
    except Exception as e:
        error_print(f"MuseTalk Error: {e}")
        error_print("如果使用MuseTalk，请先下载MuseTalk相关模型")
    gr.close_all()
    demo = main()
    demo.queue()
    
    # demo.launch()
    demo.launch(server_name=ip, # 本地端口localhost:127.0.0.1 全局端口转发:"0.0.0.0"
                server_port=port,
                # 似乎在Gradio4.0以上版本可以不使用证书也可以进行麦克风对话
                # ssl_certfile=ssl_certfile,
                # ssl_keyfile=ssl_keyfile,
                # ssl_verify=False,                
                debug=True)
