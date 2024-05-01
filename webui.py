import os
import random 
import gradio as gr
import time
from zhconv import convert
from LLM import LLM
from ASR import WhisperASR
from TFG import SadTalker 
from TTS import EdgeTTS
from src.cost_time import calculate_time

from configs import *
os.environ["GRADIO_TEMP_DIR"]= './temp'

def get_title(title = 'Linly 智能对话系统 (Linly-Talker)'):
    description = f"""
    <p style="text-align: center; font-weight: bold;">
        <span style="font-size: 28px;">{title}</span>
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
    return description

# 默认text的Example
examples =  [
    ['应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'zh-CN-XiaoxiaoNeural'],
    ['如何进行时间管理？','男性角色', 'SadTalker', 'zh-CN-YunyangNeural'],
    ['为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？','女性角色', 'SadTalker', 'zh-HK-HiuMaanNeural'],
    ['近日，苹果公司起诉高通公司，状告其未按照相关合约进行合作，高通方面尚未回应。这句话中“其”指的是谁？', '男性角色', 'SadTalker', 'zh-TW-YunJheNeural'],
    ['撰写一篇交响乐音乐会评论，讨论乐团的表演和观众的整体体验。', '男性角色', 'Wav2Lip', 'zh-CN-YunyangNeural'],
    ['翻译成中文：Luck is a dividend of sweat. The more you sweat, the luckier you get.', '女性角色', 'SadTalker', 'zh-CN-XiaoxiaoNeural'],
    ]

# 设置默认system
default_system = '你是一个很有帮助的助手'

# 设定默认参数值，可修改
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
def LLM_response(question_audio, question, voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0):
    answer = llm.generate(question)
    print(answer)
    if voice in tts.SUPPORTED_VOICE:
        try:
            tts.predict(answer, voice, rate, volume, pitch , 'answer.wav', 'answer.vtt')
        except:
            os.system(f'edge-tts --text "{answer}" --voice {voice} --write-media answer.wav')
        return 'answer.wav', 'answer.vtt', answer
    elif voice == "克隆烟嗓音":
        try:
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
            return 'answer.wav', None, answer
        except Exception as e:
            gr.Error("无克隆环境或者无克隆模型权重，无法克隆声音", e)
            return None, None, None
    elif voice == "克隆声音":
        try:
            if question_audio is None:
                gr.Error("无声音输入，无法克隆声音")
                # print("无声音输入，无法克隆声音")
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
        except Exception as e:
            gr.Error("无克隆环境或者无克隆模型权重，无法克隆声音", e)
            return None, None, None

@calculate_time
def Talker_response(question_audio = None, method = 'SadTalker', text = '', voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, batch_size = 2, character = '女性角色'):
    if character == '女性角色':
        # 女性角色
        source_image, pic_path = r'inputs/girl.png', r'inputs/girl.png'
        crop_pic_path = "./inputs/first_frame_dir_girl/girl.png"
        first_coeff_path = "./inputs/first_frame_dir_girl/girl.mat"
        crop_info = ((403, 403), (19, 30, 502, 513), [40.05956541381802, 40.17324339233366, 443.7892505041507, 443.9029284826663])
        default_voice = 'zh-CN-XiaoxiaoNeural'
    elif character == '男性角色':
        # 男性角色
        source_image = r'./inputs/boy.png'
        pic_path = "./inputs/boy.png"
        crop_pic_path = "./inputs/first_frame_dir_boy/boy.png"
        first_coeff_path = "./inputs/first_frame_dir_boy/boy.mat"
        crop_info = ((876, 747), (0, 0, 886, 838), [10.382158280494476, 0, 886, 747.7078990925525])
        default_voice = 'zh-CN-YunyangNeural'
    else:
        gr.Error('未知角色')
        return None
    voice = default_voice if voice not in tts.SUPPORTED_VOICE+["克隆烟嗓音", "克隆声音"] else voice
    print(voice, character)
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, voice, rate, volume, pitch)
    pose_style = random.randint(0, 45)
    if method == 'SadTalker':
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
    elif method == 'Wav2Lip':
        video = wav2lip.predict(crop_pic_path, driven_audio, batch_size)
    else:
        return None
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

def human_respone(history, voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0, batch_size = 2, character = '女性角色'):
    response = history[-1][1]
    driven_audio, video_vtt = 'answer.wav', 'answer.vtt'
    if character == '女性角色':
        # 女性角色
        source_image, pic_path = r'./inputs/girl.png', r"./inputs/girl.png"
        crop_pic_path = "./inputs/first_frame_dir_girl/girl.png"
        first_coeff_path = "./inputs/first_frame_dir_girl/girl.mat"
        crop_info = ((403, 403), (19, 30, 502, 513), [40.05956541381802, 40.17324339233366, 443.7892505041507, 443.9029284826663])
        default_voice = 'zh-CN-XiaoxiaoNeural'
    elif character == '男性角色':
        # 男性角色
        source_image = r'./inputs/boy.png'
        pic_path = "./inputs/boy.png"
        crop_pic_path = "./inputs/first_frame_dir_boy/boy.png"
        first_coeff_path = "./inputs/first_frame_dir_boy/boy.mat"
        crop_info = ((876, 747), (0, 0, 886, 838), [10.382158280494476, 0, 886, 747.7078990925525])
        default_voice = 'zh-CN-YunyangNeural'
    voice = default_voice if voice not in tts.SUPPORTED_VOICE else voice
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

def voice_setting(suport_voice):
    with gr.Accordion("Advanced Settings(高级设置语音参数) ", open=False):
        voice = gr.Dropdown(suport_voice,  
                            label="声音选择 Voice", 
                            value = "克隆声音" if '克隆声音' in suport_voice else None)
        rate = gr.Slider(minimum=-100,
                            maximum=100,
                            value=0,
                            step=1.0,
                            label='声音速率 Rate')
        volume = gr.Slider(minimum=0,
                                maximum=100,
                                value=100,
                                step=1,
                                label='声音音量 Volume')
        pitch = gr.Slider(minimum=-100,
                            maximum=100,
                            value=0,
                            step=1,
                            label='声音音调 Pitch')
        batch_size = gr.Slider(minimum=1,
                            maximum=10,
                            value=2,
                            step=1,
                            label='模型参数 调节可以加快生成速度 Talker Batch size')

    character = gr.Radio(['女性角色', '男性角色'], label="角色选择", value='女性角色')
    method = gr.Radio(choices = ['SadTalker', 'Wav2Lip', 'ER-NeRF(Comming Soon!!!)'], value = 'SadTalker', label = '模型选择')
    return  voice, rate, volume, pitch, batch_size, character, method

@calculate_time
def Talker_response_img(question_audio, method, text, voice, rate, volume, pitch, source_image,
                    preprocess_type, 
                    is_still_mode,
                    enhancer,
                    batch_size,                            
                    size_of_image,
                    pose_style,
                    facerender,
                    exp_weight,
                    blink_every,
                    fps):
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, voice, rate, volume, pitch)
    if method == 'SadTalker':
        video = talker.test2(source_image,
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
                        fps=fps)
    elif method == 'Wav2Lip':
        video = wav2lip.predict(source_image, driven_audio, batch_size)
    else:
        return None
    if driven_vtt:
        return video, driven_vtt
    else:
        return video

def app():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 文本/语音对话"))
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="question_audio"):
                    with gr.TabItem('对话'):
                        with gr.Column(variant='panel'):
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            voice, rate, volume, pitch, batch_size, character, method = voice_setting(tts.SUPPORTED_VOICE)
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                            asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
                        
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.TabItem('数字人问答'):
                        gen_video = gr.Video(label="生成视频", format="mp4", scale=1, autoplay=False)
                video_button = gr.Button("提交视频生成", variant='primary')
            video_button.click(fn=Talker_response,inputs=[question_audio, method, input_text,voice, rate, volume, pitch, batch_size, character],outputs=[gen_video])

        with gr.Row():
            with gr.Column(variant='panel'):
                gr.Markdown("## Test Examples")
                gr.Examples(
                    examples = examples,
                    fn = Talker_response,
                    inputs = [input_text, character, method, voice],
                )
    return inference

def app_multi():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 多轮GPT对话"))
        with gr.Row():
            with gr.Column():
                voice, rate, volume, pitch, batch_size, character, method = voice_setting(tts.SUPPORTED_VOICE)
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
                audio = gr.Audio(sources=['microphone','upload'], type="filepath", label='语音对话', autoplay=False)
                asr_text = gr.Button('🎤 语音识别（语音对话后点击）')
                
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")
                asr_text.click(fn=Asr,inputs=[audio],outputs=[msg])
                
                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    sumbit = gr.Button("🚀 发送", variant = 'primary')
                    
            # 设置按钮的点击事件。当点击时，调用上面定义的 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            sumbit.click(chat_response, inputs=[system_input, msg, chatbot], 
                         outputs=[msg, chatbot])
            
            # 点击后清空后端存储的聊天记录
            clear_history.click(fn = clear_session, outputs = [msg, chatbot])
            
            # 设置system并清除历史对话
            modify_system.click(fn=modify_system_session,
                        inputs=[system_input],
                        outputs=[system_state, system_input, chatbot])
            
            video_button.click(fn = human_respone, inputs = [chatbot, voice, rate, volume, pitch, batch_size, character], outputs = [video])
            
        with gr.Row(variant='panel'):
            with gr.Column(variant='panel'):
                gr.Markdown("## Test Examples")
                gr.Examples(
                    examples = examples,
                    fn = Talker_response,
                    inputs = [msg, character, method, voice],
                )
    return inference

def app_img():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 任意图片对话"))
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                        with gr.TabItem('Source image'):
                            with gr.Row():
                                source_image = gr.Image(label="Source image", type="filepath", elem_id="img2img_image", width=512)
                
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
                
                # with gr.Tabs(elem_id="response_audio"):
                #     with gr.TabItem("语音选择"):
                #         with gr.Column(variant='panel'):
                #             voice = gr.Dropdown(VOICES, values='zh-CN-XiaoxiaoNeural')
                            
                            
                with gr.Tabs(elem_id="text_examples"): 
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
                method = gr.Radio(choices = ['SadTalker', 'Wav2Lip', 'ER-NeRF(Comming Soon!!!)'], value = 'SadTalker', label = '模型选择')
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        with gr.Accordion("Advanced Settings",
                                        open=False):
                            gr.Markdown("SadTalker: need help? please visit our [[best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md)] for more detials")
                            with gr.Column(variant='panel'):
                                # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                                # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                                with gr.Row():
                                    pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0) #
                                    exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1) # 
                                    blink_every = gr.Checkbox(label="use eye blink", value=True)

                                with gr.Row():
                                    size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model? 256 is faster") # 
                                    preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                                
                                with gr.Row():
                                    is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                                    facerender = gr.Radio(['facevid2vid', 'PIRender'], value='facevid2vid', label='facerender', info="which face render?")
                                    
                                with gr.Row():
                                    batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1)
                                    fps = gr.Slider(label='fps in generation', step=1, maximum=30, value =20)
                                    enhancer = gr.Checkbox(label="GFPGAN as Face enhancer(slow)")                                               

                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(label="Generated video", format="mp4",scale=0.8)

                submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
            submit.click(
                fn=Talker_response_img,
                inputs=[question_audio,
                        method, 
                        input_text,
                        voice, rate, volume, pitch,
                        source_image, 
                        preprocess_type,
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        blink_every,
                        fps], 
                outputs=[gen_video]
                )
            
        with gr.Row():
            examples = [
                [
                    'examples/source_image/full_body_2.png',
                    'crop',
                    False,
                    False
                ],
                [
                    'examples/source_image/full_body_1.png',
                    'crop',
                    False,
                    False
                ],
                [
                    'examples/source_image/full3.png',
                    'crop',
                    False,
                    False
                ],
                [
                    'examples/source_image/full4.jpeg',
                    'crop',
                    False,
                    False
                ],
                [
                    'examples/source_image/art_13.png',
                    'crop',
                    False,
                    False
                ],
                [
                    'examples/source_image/art_5.png',
                    'crop',
                    False,
                    False
                ],
            ]
            gr.Examples(examples=examples,
                        fn=Talker_response,
                        inputs=[
                            source_image,
                            preprocess_type,
                            is_still_mode,
                            enhancer], 
                        outputs=[gen_video],
                        # cache_examples=True,
                        )
    return inference

def app_vits():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 语音克隆"))
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="question_audio"):
                    with gr.TabItem('对话'):
                        with gr.Column(variant='panel'):
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            voice, rate, volume, pitch, batch_size, character, method = voice_setting(["克隆声音", "克隆烟嗓音"] + tts.SUPPORTED_VOICE)
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                            asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.TabItem('数字人问答'):
                        gen_video = gr.Video(label="Generated video", format="mp4", scale=1, autoplay=False)
                video_button = gr.Button("提交", variant='primary')
            video_button.click(fn=Talker_response,inputs=[question_audio, method, input_text, voice, rate, volume, pitch, batch_size, character],outputs=[gen_video])

        with gr.Row():
            with gr.Column(variant='panel'):
                gr.Markdown("## Test Examples")
                gr.Examples(
                    examples = [["如何应对压力", "男性角色", "SadTalker", "克隆烟嗓音"], ["北京有什么好玩的地方", "男性角色", "SadTalker", "克隆声音"]] + examples,
                    fn = Talker_response,
                    inputs = [input_text, character, method, voice],
                )
    return inference
    
if __name__ == "__main__":
    # llm = LLM(mode='offline').init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf')
    # llm = LLM(mode='offline').init_model('Gemini', 'gemini-pro', api_key = "your api key")
    # llm = LLM(mode='offline').init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    llm = LLM(mode='offline').init_model('Qwen', 'Qwen/Qwen-1_8B-Chat')
    try:
        talker = SadTalker(lazy_load=True)
    except Exception as e:
        print("SadTalker Error: ", e)
        # print("如果使用SadTalker，请先下载SadTalker模型")
        gr.Warning("如果使用SadTalker，请先下载SadTalker模型")
    try:
        from TFG import Wav2Lip
        wav2lip = Wav2Lip("checkpoints/wav2lip_gan.pth")
    except Exception as e:
        print("Wav2Lip Error: ", e)
        print("如果使用Wav2Lip，请先下载Wav2Lip模型")
    try:
        from VITS import GPT_SoVITS
        vits = GPT_SoVITS()
    except Exception as e:
        print("GPT-SoVITS Error: ", e)
        print("如果使用VITS，请先下载GPT-SoVITS模型和安装环境")
    try:
        from ASR import FunASR
        asr = FunASR()
    except Exception as e:
        print("ASR Error: ", e)
        print("如果使用FunASR，请先下载FunASR模型和安装环境")
        asr = WhisperASR('base')
    tts = EdgeTTS()
    gr.close_all()
    demo_app = app()
    demo_img = app_img()
    demo_multi = app_multi()
    demo_vits = app_vits()
    demo = gr.TabbedInterface(interface_list = [demo_app, demo_img, demo_multi, demo_vits], 
                              tab_names = ["文本/语音对话", "任意图片对话", "多轮GPT对话", "语音克隆数字人对话"],
                              title = "Linly-Talker WebUI")
    demo.launch(server_name="127.0.0.1", # 本地端口localhost:127.0.0.1 全局端口转发:"0.0.0.0"
                server_port=port,
                # 似乎在Gradio4.0以上版本可以不使用证书也可以进行麦克风对话
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
                ssl_verify=False,
                debug=True,
                )