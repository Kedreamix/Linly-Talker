import os
import random 
import gradio as gr
import time
import torch
import gc
import warnings
warnings.filterwarnings('ignore')
from zhconv import convert
from LLM import LLM
from TTS import EdgeTTS
from src.cost_time import calculate_time

from configs import *
os.environ["GRADIO_TEMP_DIR"]= './temp'
os.environ["WEBUI"] = "true"
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
        <span>Linly-Talker是一款创新的数字人对话系统，它融合了最新的人工智能技术，包括大型语言模型（LLM）🤖、自动语音识别（ASR）🎙️、文本到语音转换（TTS）🗣️和语音克隆技术🎤。</span>
    </p>
    """
    return description


# 设置默认system
default_system = '你是一个很有帮助的助手'
# 设置默认的prompt
prefix_prompt = '''请用少于25个字回答以下问题\n\n'''

edgetts = EdgeTTS()

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

def clear_memory():
    """
    清理PyTorch的显存和系统内存缓存。
    """
    # 1. 清理缓存的变量
    gc.collect()  # 触发Python垃圾回收
    torch.cuda.empty_cache()  # 清理PyTorch的显存缓存
    torch.cuda.ipc_collect()  # 清理PyTorch的跨进程通信缓存
    
    # 2. 打印显存使用情况（可选）
    print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    print(f"Max cached memory: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")

@calculate_time
def TTS_response(text, 
                 voice, rate, volume, pitch,
                 am, voc, lang, male,
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, 
                 question_audio, question, use_mic_voice,
                 tts_method = 'PaddleTTS', save_path = 'answer.wav'):
    # print(text, voice, rate, volume, pitch, am, voc, lang, male, tts_method, save_path)
    if tts_method == 'Edge-TTS':
        if not edgetts.network:
            gr.Warning("请检查网络或者使用其他模型，例如PaddleTTS") 
            return None, None
        try:
            edgetts.predict(text, voice, rate, volume, pitch , 'answer.wav', 'answer.vtt')
        except:
            os.system(f'edge-tts --text "{text}" --voice {voice} --write-media answer.wav --write-subtitles answer.vtt')
        return 'answer.wav', 'answer.vtt'
    elif tts_method == 'PaddleTTS':
        tts.predict(text, am, voc, lang = lang, male=male, save_path = save_path)
        return save_path, None
    elif tts_method == 'GPT-SoVITS克隆声音':
        if use_mic_voice:
            try:
                vits.predict(ref_wav_path = question_audio,
                                prompt_text = question,
                                prompt_language = "中文",
                                text = text, # 回答
                                text_language = "中文",
                                how_to_cut = "凑四句一切",
                                save_path = 'answer.wav')
                return 'answer.wav', None
            except Exception as e:
                gr.Warning("无克隆环境或者无克隆模型权重，无法克隆声音", e)
                return None, None
        else:
            try:
                vits.predict(ref_wav_path = inp_ref,
                                prompt_text = prompt_text,
                                prompt_language = prompt_language,
                                text = text, # 回答
                                text_language = text_language,
                                how_to_cut = how_to_cut,
                                save_path = 'answer.wav')
                return 'answer.wav', None
            except Exception as e:
                gr.Warning("无克隆环境或者无克隆模型权重，无法克隆声音", e)
                return None, None
    return None, None
@calculate_time
def LLM_response(question_audio, question, 
                 voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0,
                 am='fastspeech2', voc='pwgan',lang='zh', male=False, 
                 inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", use_mic_voice = False,
                 tts_method = 'Edge-TTS'):
    answer = llm.generate(question, default_system)
    print(answer)
    driven_audio, driven_vtt = TTS_response(answer, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, question_audio, question, use_mic_voice,
                 tts_method)
    return driven_audio, driven_vtt, answer

@calculate_time
def Talker_response(question_audio = None, method = 'SadTalker', text = '',
                    voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, 
                    am = 'fastspeech2', voc = 'pwgan', lang = 'zh', male = False, 
                    inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", use_mic_voice = False,
                    tts_method = 'Edge-TTS',batch_size = 2, character = '女性角色', 
                    progress=gr.Progress(track_tqdm=True)):
    default_voice = None
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
        gr.Warning('未知角色')
        return None
    
    voice = default_voice if not voice else voice
    
    if not voice:
        gr.Warning('请选择声音')
    
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, 
                                               voice, rate, volume, pitch, 
                                               am, voc, lang, male, 
                                               inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                                               tts_method)
    if driven_audio is None:
        gr.Warning("音频没有正常生成，请检查TTS是否正确")
        return None
    if method == 'SadTalker':
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
    elif method == 'Wav2Lip':
        video = talker.predict(crop_pic_path, driven_audio, batch_size, enhancer)
    elif method == 'ER-NeRF':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + method)
        return None
    if driven_vtt:
        return video, driven_vtt
    else:
        return video

@calculate_time
def Talker_response_img(question_audio, method, text, voice, rate, volume, pitch, 
                        am, voc, lang, male, 
                        inp_ref , prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                        tts_method,
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
                        fps, progress=gr.Progress(track_tqdm=True)
                    ):
    if enhancer:
        gr.Warning("记得请先安装GFPGAN库，pip install gfpgan, 已安装可忽略")
    if not voice:
        gr.Warning("请先选择声音")
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, voice, rate, volume, pitch, 
                                               am, voc, lang, male, 
                                               inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                                               tts_method = tts_method)
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
        video = talker.predict(source_image, driven_audio, batch_size)
    elif method == 'ER-NeRF':
        video = talker.predict(driven_audio)
    else:
        return None
    if driven_vtt:
        return video, driven_vtt
    else:
        return video

@calculate_time
def Talker_Say(preprocess_type, 
                        is_still_mode,
                        enhancer,
                        batch_size,                            
                        size_of_image,
                        pose_style,
                        facerender,
                        exp_weight,
                        blink_every,
                        fps,source_image = None, source_video = None, question_audio = None, method = 'SadTalker', text = '', 
                    voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, 
                    am = 'fastspeech2', voc = 'pwgan', lang = 'zh', male = False, 
                    inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", use_mic_voice = False,
                    tts_method = 'Edge-TTS', character = '女性角色',
                    progress=gr.Progress(track_tqdm=True)):
    if source_video:
        source_image = source_video
    default_voice = None
    
    voice = default_voice if not voice else voice
    
    if not voice:
        gr.Warning('请选择声音')
    
    driven_audio, driven_vtt = TTS_response(text, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, question_audio, text, use_mic_voice,
                 tts_method)
    
    if method == 'SadTalker':
        pose_style = random.randint(0, 45)
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
        video = talker.predict(source_image, driven_audio, batch_size, enhancer)
    elif method == 'ER-NeRF':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + method)
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

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    llm.clear_history()
    return system, system, []

def clear_session():
    # clear history
    llm.clear_history()
    return '', []


def human_response(history, question_audio, talker_method, 
                   voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 0, pitch = 0, batch_size = 2, 
                  am = 'fastspeech2', voc = 'pwgan', lang = 'zh', male = False, 
                  inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", use_mic_voice = False,
                  tts_method = 'Edge-TTS', character = '女性角色', progress=gr.Progress(track_tqdm=True)):
    response = history[-1][1]
    qusetion = history[-1][0]
    # driven_audio, video_vtt = 'answer.wav', 'answer.vtt'
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
    voice = default_voice if not voice else voice
    # tts.predict(response, voice, rate, volume, pitch, driven_audio, video_vtt)
    driven_audio, driven_vtt = TTS_response(response, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 inp_ref, prompt_text, prompt_language, text_language, how_to_cut, question_audio, qusetion, use_mic_voice,
                 tts_method)
    
    if talker_method == 'SadTalker':
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
    elif talker_method == 'Wav2Lip':
        video = talker.predict(crop_pic_path, driven_audio, batch_size, enhancer)
    elif talker_method == 'ER-NeRF':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + talker_method)
        return None
    if driven_vtt:
        return video, driven_vtt
    else:
        return video


@calculate_time
def MuseTalker_response(source_video, bbox_shift, question_audio = None, text = '',
                    voice = 'zh-CN-XiaoxiaoNeural', rate = 0, volume = 100, pitch = 0, 
                    am = 'fastspeech2', voc = 'pwgan', lang = 'zh', male = False, 
                    inp_ref = None, prompt_text = "", prompt_language = "", text_language = "", how_to_cut = "", use_mic_voice = False,
                    tts_method = 'Edge-TTS', batch_size = 4,
                    progress=gr.Progress(track_tqdm=True)):
    default_voice = None    
    voice = default_voice if not voice else voice
    
    if not voice:
        gr.Warning('请选择声音')
    
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, 
                                               voice, rate, volume, pitch, 
                                               am, voc, lang, male, 
                                               inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                                               tts_method)
    print(driven_audio, driven_vtt)
    video = musetalker.inference_noprepare(driven_audio, 
                                            source_video, 
                                            bbox_shift,
                                            batch_size,
                                            fps = 25) 
    
    if driven_vtt:
        return (video, driven_vtt)
    else:
        return video 
GPT_SoVITS_ckpt = "GPT_SoVITS/pretrained_models"
def load_vits_model(gpt_path, sovits_path, progress=gr.Progress(track_tqdm=True)):
    global vits
    print("模型加载中...", gpt_path, sovits_path)
    all_gpt_path, all_sovits_path = os.path.join(GPT_SoVITS_ckpt, gpt_path), os.path.join(GPT_SoVITS_ckpt, sovits_path)
    vits.load_model(all_gpt_path, all_sovits_path)
    gr.Info("模型加载成功")
    return gpt_path, sovits_path

def list_models(dir, endwith = ".pth"):
    list_folder = os.listdir(dir)
    list_folder = [i for i in list_folder if i.endswith(endwith)]
    return list_folder

def character_change(character):
    if character == '女性角色':
        # 女性角色
        source_image = r'./inputs/girl.png'
    elif character == '男性角色':
        # 男性角色
        source_image = r'./inputs/boy.png'
    elif character == '自定义角色':
        # gr.Warnings("自定义角色暂未更新，请继续关注后续，可通过自由上传图片模式进行自定义角色")
        source_image = None
    return source_image

def webui_setting(talk = True):
    if not talk:
        with gr.Tabs():
            with gr.TabItem('数字人形象设定'):
                source_image = gr.Image(label="Source image", type="filepath")
    else:
        source_image = None
    with gr.Tabs("TTS Method"):
        with gr.Accordion("TTS Method语音方法调节 ", open=False):
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
            with gr.Tab('GPT-SoVITS'):
                with gr.Row():
                    gpt_path = gr.FileExplorer(root = GPT_SoVITS_ckpt, glob = "*.ckpt", value = "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", file_count='single', label="GPT模型路径")
                    sovits_path = gr.FileExplorer(root = GPT_SoVITS_ckpt, glob = "*.pth", value = "s2G488k.pth", file_count='single', label="SoVITS模型路径")
                    # gpt_path = gr.Dropdown(choices=list_models(GPT_SoVITS_ckpt, 'ckpt'))
                    # sovits_path = gr.Dropdown(choices=list_models(GPT_SoVITS_ckpt, 'pth'))
                    # gpt_path = gr.Textbox(label="GPT模型路径", 
                    #                       value="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
                    # sovits_path = gr.Textbox(label="SoVITS模型路径", 
                    #                          value="GPT_SoVITS/pretrained_models/s2G488k.pth")
                button = gr.Button("加载模型")
                button.click(fn = load_vits_model, 
                             inputs=[gpt_path, sovits_path], 
                             outputs=[gpt_path, sovits_path])
                
                with gr.Row():
                    inp_ref = gr.Audio(label="请上传3~10秒内参考音频，超过会报错！", sources=["microphone", "upload"], type="filepath")
                    use_mic_voice = gr.Checkbox(label="使用语音问答的麦克风")
                    prompt_text = gr.Textbox(label="参考音频的文本", value="")
                    prompt_language = gr.Dropdown(
                        label="参考音频的语种", choices=["中文", "英文", "日文"], value="中文"
                    )
                asr_button = gr.Button("语音识别 - 克隆参考音频")
                asr_button.click(fn=Asr,inputs=[inp_ref],outputs=[prompt_text])
                with gr.Row():
                    text_language = gr.Dropdown(
                        label="需要合成的语种", choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], value="中文"
                    )
                    
                    how_to_cut = gr.Dropdown(
                        label="怎么切",
                        choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切" ],
                        value="凑四句一切",
                        interactive=True,
                    )
            
            with gr.Column(variant='panel'): 
                batch_size = gr.Slider(minimum=1,
                                    maximum=10,
                                    value=2,
                                    step=1,
                                    label='Talker Batch size')

    character = gr.Radio(['女性角色', 
                          '男性角色', 
                          '自定义角色'], 
                         label="角色选择", value='自定义角色')
    # character.change(fn = character_change, inputs=[character], outputs = [source_image])
    tts_method = gr.Radio(['Edge-TTS', 'PaddleTTS', 'GPT-SoVITS克隆声音', 'Comming Soon!!!'], label="Text To Speech Method", 
                                              value = 'Edge-TTS')
    tts_method.change(fn = tts_model_change, inputs=[tts_method], outputs = [tts_method])
    asr_method = gr.Radio(choices = ['Whisper-tiny', 'Whisper-base', 'FunASR', 'Comming Soon!!!'], value='Whisper-base', label = '语音识别模型选择')
    asr_method.change(fn = asr_model_change, inputs=[asr_method], outputs = [asr_method])
    talker_method = gr.Radio(choices = ['SadTalker', 'Wav2Lip', 'ER-NeRF', 'Comming Soon!!!'], 
                      value = 'SadTalker', label = '数字人模型选择')
    talker_method.change(fn = talker_model_change, inputs=[talker_method], outputs = [talker_method])
    llm_method = gr.Dropdown(choices = ['Qwen', 'Qwen2', 'Linly', 'Gemini', 'ChatGLM', 'ChatGPT', 'GPT4Free', '直接回复 Direct Reply', 'Comming Soon!!!'], value = '直接回复 Direct Reply', label = 'LLM 模型选择')
    llm_method.change(fn = llm_model_change, inputs=[llm_method], outputs = [llm_method])
    return  (source_image, voice, rate, volume, pitch, 
             am, voc, lang, male, 
             inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
             tts_method, batch_size, character, talker_method, asr_method, llm_method)


def exmaple_setting(asr, text, character, talk , tts, voice, llm):
    # 默认text的Example
    examples =  [
        ['Whisper-base', '应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-XiaoxiaoNeural', '直接回复 Direct Reply'],
        ['Whisper-tiny', '应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'PaddleTTS', 'None', '直接回复 Direct Reply'],
        ['Whisper-base', '应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-XiaoxiaoNeural', 'Qwen'],
        ['FunASR', '如何进行时间管理？','男性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-YunyangNeural', 'Qwen'],
        ['Whisper-tiny', '为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？','女性角色', 'Wav2Lip', 'PaddleTTS', 'None', 'Qwen'],
        ]

    with gr.Row(variant='panel'):
        with gr.Column(variant='panel'):
            gr.Markdown("## Test Examples")
            gr.Examples(
                examples = examples,
                inputs = [asr, text, character, talk , tts, voice, llm],
            )
def app():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 文本/语音对话"))
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'): 
                (source_image, voice, rate, volume, pitch, 
                am, voc, lang, male, 
                inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
             
            
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem('对话'):
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                        asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
                # with gr.TabItem('SadTalker数字人参数设置'):
                #     with gr.Accordion("Advanced Settings",
                #                     open=False):
                #         gr.Markdown("SadTalker: need help? please visit our [[best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md)] for more detials")
                #         with gr.Column(variant='panel'):
                #             # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                #             # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                #             with gr.Row():
                #                 pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0) #
                #                 exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1) # 
                #                 blink_every = gr.Checkbox(label="use eye blink", value=True)

                #             with gr.Row():
                #                 size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model? 256 is faster") # 
                #                 preprocess_type = gr.Radio(['crop', 'resize','full'], value='full', label='preprocess', info="How to handle input image?")
                            
                #             with gr.Row():
                #                 is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                #                 facerender = gr.Radio(['facevid2vid'], value='facevid2vid', label='facerender', info="which face render?")
                                
                #             with gr.Row():
                #                 # batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1)
                #                 fps = gr.Slider(label='fps in generation', step=1, maximum=30, value =20)
                #                 enhancer = gr.Checkbox(label="GFPGAN as Face enhancer(slow)")       
                with gr.Tabs():
                    with gr.TabItem('数字人问答'):
                        gen_video = gr.Video(label="生成视频", format="mp4", autoplay=False)
                video_button = gr.Button("🎬 生成数字人视频", variant='primary')
            video_button.click(fn=Talker_response,inputs=[question_audio, talker_method, input_text, voice, rate, volume, pitch,
                                                          am, voc, lang, male, 
                                                          inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                                                          tts_method, batch_size, character],outputs=[gen_video])
        exmaple_setting(asr_method, input_text, character, talker_method, tts_method, voice, llm_method)
    return inference

def app_multi():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 多轮GPT对话"))
        with gr.Row():
            with gr.Column():
                (source_image, voice, rate, volume, pitch, 
                am, voc, lang, male, 
                inp_ref, prompt_text, prompt_language, text_language, how_to_cut,  use_mic_voice,
                tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
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
                with gr.Group():
                    question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label='语音对话', autoplay=False)
                    asr_text = gr.Button('🎤 语音识别（语音对话后点击）')
                
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")
                asr_text.click(fn=Asr,inputs=[question_audio],outputs=[msg])
                
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
            
            video_button.click(fn = human_response, inputs = [chatbot, question_audio, talker_method, 
                                                              voice, rate, volume, pitch, batch_size,
                                                             am, voc, lang, male, 
                                                             inp_ref, prompt_text, prompt_language, text_language, how_to_cut,  
                                                             use_mic_voice, tts_method, character], outputs = [video])
            
        exmaple_setting(asr_method, msg, character, talker_method, tts_method, voice, llm_method)
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
                                
                (_, voice, rate, volume, pitch, 
                am, voc, lang, male, 
                inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
                            
                
                    
            # driven_audio = 'answer.wav'           
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.TabItem('对话'):
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                        asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
                with gr.Tabs(elem_id="text_examples"): 
                    gr.Markdown("## Text Examples")
                    examples =  [
                        ['应对压力最有效的方法是什么？'],
                        ['如何进行时间管理？'],
                        ['为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？'],
                    ]
                    gr.Examples(
                        examples = examples,
                        inputs = [input_text],
                    )
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('SadTalker数字人参数设置'):
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
                                    facerender = gr.Radio(['facevid2vid'], value='facevid2vid', label='facerender', info="which face render?")
                                    
                                with gr.Row():
                                    batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1)
                                    fps = gr.Slider(label='fps in generation', step=1, maximum=30, value =20)
                                    enhancer = gr.Checkbox(label="GFPGAN as Face enhancer(slow)")                                               

                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(label="Generated video", format="mp4")

                submit = gr.Button('🎬 生成数字人视频', elem_id="sadtalker_generate", variant='primary')
            submit.click(
                fn=Talker_response_img,
                inputs=[question_audio,
                        talker_method, 
                        input_text,
                        voice, rate, volume, pitch,
                        am, voc, lang, male, 
                        inp_ref, prompt_text, prompt_language, text_language, how_to_cut,  use_mic_voice,
                        tts_method,
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
                    'examples/source_image/full_body_2.png', 'SadTalker',
                    'crop',
                    False,
                    False
                ],
                [
                    'examples/source_image/full_body_1.png', 'SadTalker',
                    'full',
                    True,
                    False
                ],
                [
                    'examples/source_image/full4.jpeg', 'SadTalker',
                    'crop',
                    False,
                    True
                ],
            ]
            gr.Examples(examples=examples,
                        inputs=[
                            source_image, talker_method,
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
                (source_image, voice, rate, volume, pitch, 
                am, voc, lang, male, 
                inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.TabItem('对话'):
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                        asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text])
                with gr.Tabs():
                    with gr.TabItem('数字人问答'):
                        gen_video = gr.Video(label="Generated video", format="mp4", autoplay=False)
                video_button = gr.Button("🎬 生成数字人视频", variant='primary')
            video_button.click(fn=Talker_response,inputs=[question_audio, talker_method, input_text, voice, rate, volume, pitch, am, voc, lang, male, 
                            inp_ref, prompt_text, prompt_language, text_language, how_to_cut,  use_mic_voice,
                            tts_method, batch_size, character],outputs=[gen_video])
        exmaple_setting(asr_method, input_text, character, talker_method, tts_method, voice, llm_method)
    return inference

def app_talk():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 数字人播报"))
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'): 
                with gr.Tabs():
                    with gr.Tab("图片人物"):
                        source_image = gr.Image(label='Source image', type = 'filepath')
                        
                    with gr.Tab("视频人物"):
                        source_video = gr.Video(label="Source video")
               
                (_, voice, rate, volume, pitch, 
                am, voc, lang, male, 
                inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
        
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem('对话'):
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                        asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text]) 
                with gr.Tabs():
                    with gr.TabItem('SadTalker数字人参数设置'):
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
                                    preprocess_type = gr.Radio(['crop', 'resize','full'], value='full', label='preprocess', info="How to handle input image?")
                                
                                with gr.Row():
                                    is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                                    facerender = gr.Radio(['facevid2vid'], value='facevid2vid', label='facerender', info="which face render?")
                                    
                                with gr.Row():
                                    # batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=1)
                                    fps = gr.Slider(label='fps in generation', step=1, maximum=30, value =20)
                                    enhancer = gr.Checkbox(label="GFPGAN as Face enhancer(slow)")                                               

                with gr.Tabs():
                    gen_video = gr.Video(label="Generated video", format="mp4")

                video_button = gr.Button('🎬 生成数字人视频', elem_id="sadtalker_generate", variant='primary')

                video_button.click(fn=Talker_Say,inputs=[preprocess_type, is_still_mode, enhancer, batch_size, size_of_image,
                                pose_style, facerender, exp_weight, blink_every, fps,
                                source_image, source_video, question_audio, talker_method, input_text, voice, rate, volume, pitch, am, voc, lang, male, 
                                inp_ref, prompt_text, prompt_language, text_language, how_to_cut,  use_mic_voice,
                                tts_method, character],outputs=[gen_video])
            
        with gr.Row():
            with gr.Column(variant='panel'):
                gr.Markdown("## Test Examples")
                gr.Examples(
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
                    ],
                    fn = Talker_Say,
                    inputs = [source_image, input_text],
                )   
    return inference

def load_musetalk_model():
    gr.Info("MuseTalk模型导入中...")
    musetalker.init_model()
    gr.Info("MuseTalk模型导入成功")
    return "MuseTalk模型导入成功"
def musetalk_prepare_material(source_video, bbox_shift):
    if musetalker.load is False:
        gr.Warning("请先加载MuseTalk模型后重新上传文件")
        return source_video, None
    return musetalker.prepare_material(source_video, bbox_shift)
def app_muse():
    with gr.Blocks(analytics_enabled=False, title = 'Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) MuseTalker数字人实时对话"))
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'): 
                with gr.TabItem('MuseV Video'):
                    gr.Markdown("MuseV: need help? please visit MuseVDemo to generate Video https://huggingface.co/spaces/AnchorFake/MuseVDemo")
                    with gr.Row():
                        source_video = gr.Video(label="Reference Video",sources=['upload'])
                    gr.Markdown("BBox_shift 推荐值下限，在生成初始结果后生成相应的 bbox 范围。如果结果不理想，可以根据该参考值进行调整。\n一般来说，在我们的实验观察中，我们发现正值（向下半部分移动）通常会增加嘴巴的张开度，而负值（向上半部分移动）通常会减少嘴巴的张开度。然而，需要注意的是，这并不是绝对的规则，用户可能需要根据他们的具体需求和期望效果来调整该参数。")
                    with gr.Row():
                        bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
                        bbox_shift_scale = gr.Textbox(label="bbox_shift_scale", 
                                                        value="",interactive=False)
                load_musetalk = gr.Button("加载MuseTalk模型(传入视频前先加载)", variant='primary')
                load_musetalk.click(fn=load_musetalk_model, outputs=bbox_shift_scale)

                (_, voice, rate, volume, pitch, 
                am, voc, lang, male, 
                inp_ref, prompt_text, prompt_language, text_language, how_to_cut, use_mic_voice,
                tts_method, batch_size, character, talker_method, asr_method, llm_method)= webui_setting()
            
            source_video.change(fn=musetalk_prepare_material, inputs=[source_video, bbox_shift], outputs=[source_video, bbox_shift_scale])
            
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem('对话'):
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label = '语音对话')
                            input_text = gr.Textbox(label="Input Text", lines=3)
                            asr_text = gr.Button('语音识别（语音对话后点击）')
                        asr_text.click(fn=Asr,inputs=[question_audio],outputs=[input_text]) 
            
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
                fn=MuseTalker_response,
                inputs=[source_video, bbox_shift, question_audio, input_text, voice, rate, volume, pitch, am, voc, lang, male, 
                            inp_ref, prompt_text, prompt_language, text_language, how_to_cut,  use_mic_voice,
                            tts_method, batch_size], 
                outputs=[gen_video]
                )
    return inference
def asr_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global asr

    # 清理显存，在加载新的模型之前释放不必要的显存
    clear_memory()

    if model_name == "Whisper-tiny":
        try:
            asr = WhisperASR('tiny')
            # asr = WhisperASR('Whisper/tiny.pt')
            gr.Info("Whisper-tiny模型导入成功")
        except Exception as e:
            gr.Warning(f"Whisper-tiny模型下载失败 {e}")
    elif model_name == "Whisper-base":
        try:
            asr = WhisperASR('base')
            # asr = WhisperASR('Whisper/base.pt')
            gr.Info("Whisper-base模型导入成功")
        except Exception as e:
            gr.Warning(f"Whisper-base模型下载失败 {e}")
    elif model_name == 'FunASR':
        try:
            from ASR import FunASR
            asr = FunASR()
            gr.Info("FunASR模型导入成功")
        except Exception as e:
            gr.Warning(f"FunASR模型下载失败 {e}")
    else:
        gr.Warning("未知ASR模型，可提issue和PR 或者 建议更新模型")
    return model_name

def llm_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global llm
    gemini_apikey = ""
    openai_apikey = ""
    proxy_url = None

    # 清理显存，在加载新的模型之前释放不必要的显存
    clear_memory()

    if model_name == 'Linly':
        try:
            llm = llm_class.init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf', prefix_prompt=prefix_prompt)
            gr.Info("Linly模型导入成功")
        except Exception as e:
            gr.Warning(f"Linly模型下载失败 {e}")
    elif model_name == 'Qwen':
        try:
            llm = llm_class.init_model('Qwen', 'Qwen/Qwen-1_8B-Chat', prefix_prompt=prefix_prompt)
            gr.Info("Qwen模型导入成功")
        except Exception as e:
            gr.Warning(f"Qwen模型下载失败 {e}")
    elif model_name == 'Qwen2':
        try:
            llm = llm_class.init_model('Qwen2', 'Qwen/Qwen1.5-0.5B-Chat', prefix_prompt=prefix_prompt)
            gr.Info("Qwen2模型导入成功")
        except Exception as e:
            gr.Warning(f"Qwen2模型下载失败 {e}")
    elif model_name == 'Gemini':
        if gemini_apikey:
            llm = llm_class.init_model('Gemini', 'gemini-pro', gemini_apikey, proxy_url)
            gr.Info("Gemini模型导入成功")
        else:
            gr.Warning("请填写Gemini的api_key")
    elif model_name == 'ChatGLM':
        try:
            llm = llm_class.init_model('ChatGLM', 'THUDM/chatglm3-6b', prefix_prompt=prefix_prompt)
            gr.Info("ChatGLM模型导入成功")
        except Exception as e:
            gr.Warning(f"ChatGLM模型导入失败 {e}")
    elif model_name == 'ChatGPT':
        if openai_apikey:
            llm = llm_class.init_model('ChatGPT', api_key=openai_apikey, proxy_url=proxy_url, prefix_prompt=prefix_prompt)
        else:
            gr.Warning("请填写OpenAI的api_key")
    elif model_name == '直接回复 Direct Reply':
        llm =llm_class.init_model(model_name)
        gr.Info("直接回复，不实用LLM模型")
    elif model_name == 'GPT4Free':
        try:
            llm = llm_class.init_model('GPT4Free', prefix_prompt=prefix_prompt)
            gr.Info("GPT4Free模型导入成功, 请注意GPT4Free可能不稳定")
        except Exception as e:
            gr.Warning(f"GPT4Free模型下载失败 {e}")
    else:
        gr.Warning("未知LLM模型，可提issue和PR 或者 建议更新模型")
    return model_name
    
def talker_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global talker

    # 清理显存，在加载新的模型之前释放不必要的显存
    clear_memory()

    if model_name not in ['SadTalker', 'Wav2Lip', 'ER-NeRF']:
        gr.Warning("其他模型还未集成，请等待")
    if model_name == 'SadTalker':
        try:
            from TFG import SadTalker
            talker = SadTalker(lazy_load=True)
            gr.Info("SadTalker模型导入成功")
        except Exception as e:
            gr.Warning("SadTalker模型下载失败", e)
    elif model_name == 'Wav2Lip':
        try:
            from TFG import Wav2Lip
            clear_memory()
            talker = Wav2Lip("checkpoints/wav2lip_gan.pth")
            gr.Info("Wav2Lip模型导入成功")
        except Exception as e:
            gr.Warning("Wav2Lip模型下载失败", e)
    elif model_name == 'ER-NeRF':
        try:
            from TFG import ERNeRF
            talker = ERNeRF()
            talker.init_model('checkpoints/Obama_ave.pth', 'checkpoints/Obama.json')
            gr.Info("ER-NeRF模型导入成功")
        except Exception as e:
            gr.Warning("ER-NeRF模型下载失败", e)
    else:
        gr.Warning("未知TFG模型，可提issue和PR 或者 建议更新模型")
    return model_name

def tts_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    global tts

    # 清理显存，在加载新的模型之前释放不必要的显存
    clear_memory()

    if model_name == 'Edge-TTS':
        # tts = EdgeTTS()
        if edgetts.network:
            gr.Info("EdgeTTS模型导入成功")
        else:
            gr.Warning("EdgeTTS模型加载失败，请检查网络是否正常连接，否则无法使用")
    elif model_name == 'PaddleTTS':
        try:
            from TTS import PaddleTTS
            tts = PaddleTTS()
            gr.Info("PaddleTTS模型导入成功")
        except Exception as e:
            gr.Warning(f"PaddleTTS模型下载失败 {e}")
    elif model_name == 'GPT-SoVITS克隆声音':
        try:
            gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
            vits.load_model(gpt_path, sovits_path)
            gr.Info("模型加载成功")
        except Exception as e:
            gr.Warning(f"模型加载失败 {e}")
        gr.Warning("注意注意⚠️：GPT-SoVITS要上传参考音频进行克隆，请点击TTS Method语音方法调节操作")
    else:
        gr.Warning("未知TTS模型，可提issue和PR 或者 建议更新模型")
    return model_name

def success_print(text):
    print(f"\033[1;32;40m{text}\033[0m")

def error_print(text):
    print(f"\033[1;31;40m{text}\033[0m")

if __name__ == "__main__":
    llm_class = LLM(mode='offline')
    llm = llm_class.init_model('直接回复 Direct Reply')
    success_print("默认不使用LLM模型，直接回复问题，同时减少显存占用！")
    
    try:
        from VITS import *
        vits = GPT_SoVITS()
        success_print("Success!!! GPT-SoVITS模块加载成功，语音克隆默认使用GPT-SoVITS模型")
        # gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        # sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        # vits.load_model(gpt_path, sovits_path)
    except Exception as e:
        error_print(f"GPT-SoVITS Error: {e}")
        error_print("如果使用VITS，请先下载GPT-SoVITS模型和安装环境")
    
    try:
        from TFG import SadTalker
        talker = SadTalker(lazy_load=True)
        success_print("Success!!! SadTalker模块加载成功，默认使用SadTalker模型")
    except Exception as e:
        error_print(f"SadTalker Error: {e}")
        error_print("如果使用SadTalker，请先下载SadTalker模型")
    
    try:
        from ASR import WhisperASR
        asr = WhisperASR('base')
        success_print("Success!!! WhisperASR模块加载成功，默认使用Whisper-base模型")
    except Exception as e:
        error_print(f"ASR Error: {e}")
        error_print("如果使用FunASR，请先下载WhisperASR模型和安装环境")
    
    # 判断显存是否8g，若小于8g不建议使用MuseTalk功能
    # Check if GPU is available and has at least 8GB of memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        if gpu_memory < 8:
            error_print("警告: 您的显卡显存小于8GB，不建议使用MuseTalk功能")
    
    try:
        from TFG import MuseTalk_RealTime
        musetalker = MuseTalk_RealTime()
        success_print("Success!!! MuseTalk模块加载成功")
    except Exception as e:
        error_print(f"MuseTalk Error: {e}")
        error_print("如果使用MuseTalk，请先下载MuseTalk模型")

    tts = edgetts
    if not tts.network:
        error_print("EdgeTTS模块加载失败，请检查网络是否正常连接，否则无法使用")

    gr.close_all()
    demo_app = app()
    demo_img = app_img()
    demo_multi = app_multi()
    demo_vits = app_vits()
    demo_talk = app_talk()
    demo_muse = app_muse()
    demo = gr.TabbedInterface(interface_list = [demo_app, 
                                                demo_img, 
                                                demo_multi, 
                                                demo_vits, 
                                                demo_talk,
                                                demo_muse,
                                                ], 
                              tab_names = ["文本/语音对话", 
                                           "任意图片对话", 
                                           "多轮GPT对话", 
                                           "语音克隆数字人对话", 
                                           "数字人文本/语音播报",
                                           "MuseTalk数字人实时对话"
                                           ],
                              title = "Linly-Talker WebUI")
    demo.queue()
    demo.launch(server_name=ip, # 本地端口localhost:127.0.0.1 全局端口转发:"0.0.0.0"
                server_port=port,
                # 似乎在Gradio4.0以上版本可以不使用证书也可以进行麦克风对话
                # ssl_certfile=ssl_certfile,
                # ssl_keyfile=ssl_keyfile,
                # ssl_verify=False,
                # share=True,
                debug=True,
                )