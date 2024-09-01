import os
import random 
import gradio as gr

from src.cost_time import calculate_time

from configs import *
os.environ["GRADIO_TEMP_DIR"]= './temp'

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

# 设定默认参数值，可修改
# source_image = r'example.png'
blink_every = True
size_of_image = 256
preprocess_type = 'crop'
facerender = 'facevid2vid'
enhancer = False
is_still_mode = False

exp_weight = 1

use_ref_video = False
ref_video = None
ref_info = 'pose'
use_idle_mode = False
length_of_audio = 5

@calculate_time
def TTS_response(text, 
                 voice, rate, volume, pitch,
                 am, voc, lang, male,
                 tts_method = 'PaddleTTS', save_path = 'answer.wav'):
    print(text, voice, rate, volume, pitch, am, voc, lang, male, tts_method, save_path)
    if tts_method == 'Edge-TTS':
        try:
            edgetts.predict(text, voice, rate, volume, pitch , 'answer.wav', 'answer.vtt')
        except:
            os.system(f'edge-tts --text "{text}" --voice {voice} --write-media answer.wav')
        return 'answer.wav'
    elif tts_method == 'PaddleTTS':
        paddletts.predict(text, am, voc, lang = lang, male=male, save_path = save_path)
        return save_path

@calculate_time
def Talker_response(source_image, source_video, method = 'SadTalker', driven_audio = '', batch_size = 2):
    # print(source_image, method , driven_audio, batch_size)
    if source_video:
        source_image = source_video
    print(source_image, method , driven_audio, batch_size)
    pose_style = random.randint(0, 45)
    if method == 'SadTalker':
        video = sadtalker.test2(source_image,
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
    elif method == 'Wav2Lip':
        video = wav2lip.predict(source_image, driven_audio, batch_size)
    elif method == 'NeRFTalk':
        video = nerftalk.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + method)
        return None
    return video

def main():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(description)
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.Tab("图片人物"):
                        source_image = gr.Image(label='Source image', type = 'filepath')
                        
                    with gr.Tab("视频人物"):
                        source_video = gr.Video(label="Source video")
                
                with gr.Tabs():
                    input_audio = gr.Audio(sources=['upload', 'microphone'], type="filepath", label = '语音')
                    input_text = gr.Textbox(label="Input Text", lines=3)
                    with gr.Column():
                        tts_method = gr.Radio(["Edge-TTS", "PaddleTTS"], label="Text To Speech Method (Edge-TTS利用微软的TTS，PaddleSpeech是离线的TTS，不过第一次运行会自动下载模型)", 
                                              value = 'Edge-TTS')
                        
                with gr.Tabs("TTS Method"):
                    # with gr.Accordion("Advanced Settings(高级设置语音参数) ", open=False):
                    with gr.Tab("Edge-TTS"):
                        
                        voice = gr.Dropdown(edgetts.SUPPORTED_VOICE, 
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
                    with gr.Tab("PaddleTTS"):
                        am = gr.Dropdown(["FastSpeech2"], label="声学模型选择", value = 'FastSpeech2')
                        voc = gr.Dropdown(["PWGan", "HifiGan"], label="声码器选择", value = 'PWGan')
                        lang = gr.Dropdown(["zh", "en", "mix", "canton"], label="语言选择", value = 'zh')
                        male = gr.Checkbox(label="男声(Male)", value=False)
                    with gr.Column(variant='panel'): 
                        batch_size = gr.Slider(minimum=1,
                                            maximum=10,
                                            value=2,
                                            step=1,
                                            label='Talker Batch size')
                button_text = gr.Button('语音生成')
                button_text.click(fn=TTS_response,inputs=[input_text, voice, rate, volume, pitch, am, voc, lang, male, tts_method],
                                  outputs=[input_audio])
                        
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.TabItem('数字人问答'):
                        method = gr.Radio(choices = ['SadTalker', 'Wav2Lip', 'NeRFTalk'], value = 'SadTalker', label = '模型选择')
                        gen_video = gr.Video(label="Generated video", format="mp4", scale=1, autoplay=True)
                video_button = gr.Button("提交", variant='primary')

            video_button.click(fn=Talker_response,inputs=[source_image, source_video, method, input_audio, batch_size] ,
                               outputs=[gen_video])
            
        with gr.Row():
            examples = [
                [
                    'examples/source_image/full_body_2.png',
                    '应对压力最有效的方法是什么？',
                ],
                [
                    'examples/source_image/full_body_1.png',
                    '如何进行时间管理？',
                ],
                [
                    'examples/source_image/full3.png',
                    '为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？',
                ],
                [
                    'examples/source_image/full4.jpeg',
                    '近日，苹果公司起诉高通公司，状告其未按照相关合约进行合作，高通方面尚未回应。这句话中“其”指的是谁？',
                ],
                [
                    'examples/source_image/art_13.png',
                    '三年级同学种树80颗，四、五年级种的棵树比三年级种的2倍多14棵，三个年级共种树多少棵?',
                ],
                [
                    'examples/source_image/art_5.png',
                    '撰写一篇交响乐音乐会评论，讨论乐团的表演和观众的整体体验。',
                ],
            ]
            gr.Examples(examples=examples,
                        inputs=[
                            source_image,
                            input_text,
                            ], 
                        )
    return inference
    
if __name__ == "__main__":
    try:
        from TFG import SadTalker
        sadtalker = SadTalker(lazy_load=True)
    except Exception as e:
        print("SadTalker Error: ", e)
        print("如果使用SadTalker，请先下载SadTalker模型")

    try:
        from TFG import Wav2Lip
        wav2lip = Wav2Lip("checkpoints/wav2lip_gan.pth")
    except Exception as e:
        print("Wav2Lip Error: ", e)
        print("如果使用Wav2Lip，请先下载Wav2Lip模型")
        
    try:
        from TFG import NeRFTalk
        nerftalk = NeRFTalk()
        nerftalk.init_model('checkpoints/Obama_ave.pth', 'checkpoints/Obama.json')
    except Exception as e:
        print("ERNeRF Error: ", e)
        print("如果使用ERNeRF，请先下载ERNeRF模型")
    
    try:
        from TTS import EdgeTTS
        edgetts = EdgeTTS()
    except Exception as e:
        print("EdgeTTS Error: ", e)
        print("如果使用EdgeTTS，请先下载EdgeTTS模型")
    
    try:
        from TTS import PaddleTTS
        paddletts = PaddleTTS()
    except Exception as e:
        print("PaddleTTS Error: ", e)
        print("如果使用PaddleTTS，请先下载PaddleTTS模型")
        
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