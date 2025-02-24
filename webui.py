import os
import random 
import gradio as gr
import numpy as np
import time
import torch, torchaudio
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

# Default system and prompt settings
DEFAULT_SYSTEM = '你是一个很有帮助的助手'
PREFIX_PROMPT = '请用少于25个字回答以下问题\n\n'
# Default parameters
IMAGE_SIZE = 256
PREPROCESS_TYPE = 'crop'
FACERENDER = 'facevid2vid'
ENHANCER = False
IS_STILL_MODE = False
EXP_WEIGHT = 1
USE_REF_VIDEO = False
REF_VIDEO = None
REF_INFO = 'pose'
USE_IDLE_MODE = False
AUDIO_LENGTH = 5

edgetts = EdgeTTS()

@calculate_time
def Asr(audio):
    try:
        question = asr.transcribe(audio)
        question = convert(question, 'zh-cn')
    except Exception as e:
        gr.Warning("ASR Error: ", e)
        question = 'Gradio存在一些bug，麦克风模式有时候可能音频还未传入，请重新点击一下语音识别即可'
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

def generate_seed():
    seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": seed}

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def change_instruction(mode):
    return instruct_dict.get(mode, '未知模式')

PROMPT_SR, TARGET_SR = 16000, 22050
DEFAULT_DATA = np.zeros(TARGET_SR)

@calculate_time
def TTS_response(text, voice, rate, volume, pitch, am, voc, lang, male,
                ref_audio, prompt_text, prompt_language, text_language,
                cut_method, question_audio, question, use_mic_voice,
                mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, 
                tts_method='Edge-TTS', save_path='answer.wav'):
    if text == '':
        text = '请输入文字/问题'
    if tts_method == 'Edge-TTS':
        if not edgetts.network:
            gr.Warning("请检查网络或使用其他模型，例如PaddleTTS")
            return None
        try:
            edgetts.predict(text, voice, rate, volume, pitch, save_path, 'answer.vtt')
        except Exception as e:
            os.system(f'edge-tts --text "{text}" --voice {voice} --write-media {save_path} --write-subtitles answer.vtt')
        return save_path
    
    if tts_method == 'PaddleTTS':
        tts.predict(text, am, voc, lang=lang, male=male, save_path=save_path)
        return save_path
    
    if tts_method == 'GPT-SoVITS克隆声音':
        try:
            vits.predict(ref_wav_path=question_audio if use_mic_voice else ref_audio,
                         prompt_text=question if use_mic_voice else prompt_text,
                         prompt_language=prompt_language,
                         text=text,
                         text_language=text_language,
                         how_to_cut=cut_method,
                         save_path=save_path)
            return save_path
        except Exception as e:
            gr.Warning("无克隆环境或模型权重，无法克隆声音", e)
            return None
    elif "CosyVoice" in tts_method:
        if prompt_wav_upload is not None:
            prompt_wav = prompt_wav_upload
        elif prompt_wav_record is not None:
            prompt_wav = prompt_wav_record
        else:
            prompt_wav = None
        if mode_checkbox_group in ['跨语种复刻']:
            if prompt_wav is None:
                gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
                return (TARGET_SR, DEFAULT_DATA)
            gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
        # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
        if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
            if prompt_wav is None:
                gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
                return (TARGET_SR, DEFAULT_DATA)
            if torchaudio.info(prompt_wav).sample_rate < PROMPT_SR:
                gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, PROMPT_SR))
                return (TARGET_SR, DEFAULT_DATA)
        # sft mode only use sft_dropdown
        if mode_checkbox_group in ['预训练音色']:
            if prompt_wav is not None or prompt_text_cv != '':
                gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
        # zero_shot mode only use prompt_wav prompt text
        if mode_checkbox_group in ['3s极速复刻']:
            if prompt_text_cv == '':
                gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
                return (TARGET_SR, DEFAULT_DATA)
            # if instruct_text != '':
            #     gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

        if mode_checkbox_group == '预训练音色':
            set_all_random_seed(seed)
            output = cosyvoice.predict_sft(text, sft_dropdown, speed_factor=speed_factor, save_path=save_path)
        elif mode_checkbox_group == '3s极速复刻':
            set_all_random_seed(seed)
            output = cosyvoice.predict_zero_shot(text, prompt_text_cv, prompt_wav, speed_factor=speed_factor, save_path=save_path)
        elif mode_checkbox_group == '跨语种复刻':
            set_all_random_seed(seed)
            output = cosyvoice.predict_cross_lingual(text, prompt_wav, speed_factor=speed_factor, save_path=save_path)
        return output
    else:
        gr.Warning('未知模型')
    return None

inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2. 点击生成音频按钮',
                '3s极速复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 输入prompt文本\n3. 点击生成音频按钮',
                '跨语种复刻': '1. 选择prompt音频文件，或录入prompt音频，注意不超过30s，若同时提供，优先选择prompt音频文件\n2. 点击生成音频按钮',
                '自然语言控制': '1. 选择预训练音色\n2. 输入instruct文本\n3. 点击生成音频按钮'}


@calculate_time
def LLM_response(
    question_audio, question,  # 输入的音频和文本问题
    voice, rate, volume, pitch,  # 语音合成参数
    am, voc, lang, male,  # TTS 模型参数
    ref_audio, prompt_text, prompt_language, text_language,  # 提示音频、文本及其语言设置
    cut_method, use_mic_voice, mode_checkbox_group, sft_dropdown,  # 其他TTS选项
    prompt_text_cv, prompt_wav_upload, prompt_wav_record,  # 提示信息和音频选项
    seed, speed_factor,  # 随机种子和语速因子
    tts_method='Edge-TTS'  # TTS 方法，默认使用 'Edge-TTS'
):
    if len(question) == 0:
        gr.Warning("请输入问题")
        return None, None, None

    # 生成回答
    answer = llm.generate(question, DEFAULT_SYSTEM)
    print("LLM 回复：", answer)

    # 合成回答语音
    tts_audio = TTS_response(
        answer, voice, rate, volume, pitch, am, voc, lang, male,
        ref_audio, prompt_text, prompt_language, text_language, 
        cut_method, question_audio, question, use_mic_voice, 
        mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, 
        prompt_wav_record, seed, speed_factor, tts_method
    )

    # 生成VTT文件（如果TTS方法为'Edge-TTS'）
    tts_vtt = 'answer.vtt' if tts_method == 'Edge-TTS' else None
    tts_vtt = None
    return tts_audio, tts_vtt, answer

@calculate_time
def Talker_response_img(question_audio, method, text, voice, rate, volume, pitch,
                        am, voc, lang, male, inp_ref, prompt_text, prompt_language,
                        text_language, how_to_cut, use_mic_voice, 
                        mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, 
                        tts_method, source_image, preprocess_type, is_still_mode, enhancer,
                        batch_size, size_of_image, pose_style, facerender,
                        exp_weight, blink_every, fps, progress=gr.Progress(track_tqdm=True)):

    if enhancer:
        gr.Warning("请先安装GFPGAN库 (pip install gfpgan)，已安装可忽略")

    if not voice:
        gr.Warning("请选择声音")
        return None
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, voice, rate, volume, pitch,
                                            am, voc, lang, male, inp_ref, prompt_text, prompt_language,
                                            text_language, how_to_cut, use_mic_voice, 
                                            mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, tts_method)

    if driven_audio is None:
        gr.Warning("音频没有正常生成，请检查TTS是否正确")
        return None

    # 视频生成
    video = None
    if method == 'SadTalker':
        video = talker.test2(source_image, driven_audio, preprocess_type, is_still_mode, enhancer,
                             batch_size, size_of_image, pose_style, facerender, exp_weight,
                             REF_VIDEO, REF_INFO, USE_IDLE_MODE, AUDIO_LENGTH, blink_every, 
                             fps=fps)
    elif method == 'Wav2Lip':
        video = talker.predict(source_image, driven_audio, batch_size)
    elif method == 'Wav2Lipv2':
        video = talker.run(source_image, driven_audio, batch_size)
    elif method == 'NeRFTalk':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + method)
        return None

    return (video, driven_vtt) if driven_vtt else video

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
        system = DEFAULT_SYSTEM
    llm.clear_history()
    return system, system, []

def clear_session():
    # clear history
    llm.clear_history()
    return '', []


def human_response(source_image, history, question_audio, talker_method, voice, rate, volume, pitch,
                   am, voc, lang, male, inp_ref, prompt_text, prompt_language, text_language, cut_method, use_mic_voice, 
                   mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, 
                   tts_method, character, preprocess_type, is_still_mode,
                   enhancer, batch_size, size_of_image, pose_style, facerender, exp_weight,
                   blink_every, fps=20, progress=gr.Progress(track_tqdm=True)):
    response = history[-1][1]
    question = history[-1][0]

    # 角色信息设置
    if character == '女性角色':
        source_image = pic_path = crop_pic_path = first_coeff_path = r'./inputs/girl.png'
        crop_info = ((403, 403), (19, 30, 502, 513), [40.06, 40.17, 443.79, 443.90])
        default_voice = 'zh-CN-XiaoxiaoNeural'
    elif character == '男性角色':
        source_image = pic_path = crop_pic_path = first_coeff_path = r'./inputs/boy.png'
        crop_info = ((876, 747), (0, 0, 886, 838), [10.38, 0, 886, 747.71])
        default_voice = 'zh-CN-YunyangNeural'
    elif character == '自定义角色':
        if source_image is None:
            gr.Error("自定义角色需要上传正确的图片")
            return None
        default_voice = 'zh-CN-XiaoxiaoNeural'
    else:
        gr.Error("未知角色")
        return None

    voice = default_voice if not voice else voice

    # TTS响应生成
    driven_audio = TTS_response(response, voice, rate, volume, pitch, am, voc, lang, male,
                                            inp_ref, prompt_text, prompt_language, text_language,
                                            cut_method, question_audio, question, use_mic_voice, 
                                            mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor,tts_method)
    driven_vtt = 'answer.vtt' if tts_method == 'Edge-TTS' else None
    driven_vtt = None
    if driven_audio is None:
        gr.Warning("音频没有正常生成，请检查TTS是否正确")
        return None

    # 视频生成
    video = None
    if talker_method == 'SadTalker':
        pose_style = random.randint(0, 45)
        video = talker.test2(source_image, driven_audio, preprocess_type, is_still_mode, enhancer,
                        batch_size, size_of_image, pose_style, facerender, exp_weight,
                        REF_VIDEO, REF_INFO, USE_IDLE_MODE, AUDIO_LENGTH, blink_every, 
                        fps=fps)
    elif talker_method == 'Wav2Lip':
        video = talker.predict(crop_pic_path, driven_audio, batch_size, enhancer)
    elif talker_method == 'Wav2Lipv2':
        video = talker.run(crop_pic_path, driven_audio, batch_size, enhancer)
    elif talker_method == 'NeRFTalk':
        video = talker.predict(driven_audio)
    else:
        gr.Warning("不支持的方法：" + talker_method)
        return None

    return video, driven_vtt if driven_vtt else video


@calculate_time
def MuseTalker_response(source_video, bbox_shift, question_audio, text, voice,
                        rate, volume, pitch, am, voc, lang, male, 
                        ref_audio, prompt_text, prompt_language, text_language, cut_method, use_mic_voice,
                        mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor,  
                        tts_method='Edge-TTS', batch_size=4, progress=gr.Progress(track_tqdm=True)):
    default_voice = None
    voice = default_voice if not voice else voice

    if not voice:
        gr.Warning('请选择声音')
        return None

    # LLM响应生成
    driven_audio, driven_vtt, _ = LLM_response(question_audio, text, voice, rate, volume, pitch,
                                               am, voc, lang, male, ref_audio, prompt_text, prompt_language,
                                               text_language, cut_method, use_mic_voice, 
                                               mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, 
                                               tts_method)

    if driven_audio is None:
        gr.Warning("音频没有正常生成，请检查TTS是否正确")
        return None

    # MuseTalker 视频生成
    video = musetalker.inference_noprepare(driven_audio, source_video, bbox_shift, batch_size, fps=25)

    return (video, driven_vtt) if driven_vtt else video

GPT_SoVITS_ckpt = "GPT_SoVITS/pretrained_models"
def load_vits_model(gpt_path, sovits_path, progress=gr.Progress(track_tqdm=True)):
    global vits
    print("模型加载中...", gpt_path, sovits_path)
    all_gpt_path = os.path.join(GPT_SoVITS_ckpt, gpt_path)
    all_sovits_path = os.path.join(GPT_SoVITS_ckpt, sovits_path)
    vits.load_model(all_gpt_path, all_sovits_path)
    gr.Info("模型加载成功")
    return gpt_path, sovits_path

def character_change(character):
    if character == '女性角色':
        return r'./inputs/girl.png'
    elif character == '男性角色':
        return r'./inputs/boy.png'
    elif character == '自定义角色':
        return None
    else:
        gr.Warning("不支持的角色类型：" + character)
        return None

def webui_setting(talk=False):
    if not talk:
        with gr.Tabs():
            with gr.TabItem('数字人形象设定'):
                source_image = gr.Image(label="Source image", type="filepath")
    else:
        source_image = None
    with gr.Tabs("TTS Method"):
        with gr.Accordion("TTS Method语音方法调节 ", open=True):
            with gr.Tab("Edge-TTS"):
                voice = gr.Dropdown(edgetts.SUPPORTED_VOICE, value='zh-CN-XiaoxiaoNeural', label="Voice 声音选择")
                rate = gr.Slider(minimum=-100, maximum=100, value=0, step=1.0, label='Rate 速率')
                volume = gr.Slider(minimum=0, maximum=100, value=100, step=1, label='Volume 音量')
                pitch = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label='Pitch 音调')
            with gr.Tab("PaddleTTS"):
                am = gr.Dropdown(["FastSpeech2"], label="声学模型选择", value='FastSpeech2')
                voc = gr.Dropdown(["PWGan", "HifiGan"], label="声码器选择", value='PWGan')
                lang = gr.Dropdown(["zh", "en", "mix", "canton"], label="语言选择", value='zh')
                male = gr.Checkbox(label="男声(Male)", value=False)
            with gr.Tab('GPT-SoVITS'):
                with gr.Row():
                    gpt_path = gr.FileExplorer(root=GPT_SoVITS_ckpt, glob="*.ckpt", value="s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", file_count='single', label="GPT模型路径")
                    sovits_path = gr.FileExplorer(root=GPT_SoVITS_ckpt, glob="*.pth", value="s2G488k.pth", file_count='single', label="SoVITS模型路径")
                button = gr.Button("加载模型")
                button.click(fn=load_vits_model, inputs=[gpt_path, sovits_path], outputs=[gpt_path, sovits_path])
                with gr.Row():
                    ref_audio = gr.Audio(label="请上传3~10秒内参考音频，超过会报错！", sources=["microphone", "upload"], type="filepath")
                    use_mic_voice = gr.Checkbox(label="使用语音问答的麦克风")
                    prompt_text = gr.Textbox(label="参考音频的文本", value="")
                    prompt_language = gr.Dropdown(label="参考音频的语种", choices=["中文", "英文", "日文"], value="中文")
                asr_button = gr.Button("语音识别 - 克隆参考音频")
                asr_button.click(fn=Asr, inputs=[ref_audio], outputs=[prompt_text])
                with gr.Row():
                    text_language = gr.Dropdown(label="需要合成的语种", choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], value="中文")
                    cut_method = gr.Dropdown(label="怎么切", choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"], value="凑四句一切", interactive=True)
            
            with gr.Tab('CosyVoice'):
                # tts_text = gr.Textbox(label="输入合成文本", lines=1, value="我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。")
                speed_factor = gr.Slider(minimum=0.25, maximum=4, step=0.05, label="语速调节", value=1.0, interactive=True)
                with gr.Row():
                    mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='选择推理模式', value=inference_mode_list[0])
                    instruction_text = gr.Text(label="操作步骤", lines=3, value=instruct_dict[inference_mode_list[0]], scale=0.5)
                    sft_dropdown = gr.Dropdown(choices=['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女'], label='选择预训练音色', value="中文女", scale=0.25)
                with gr.Row():
                    seed_button = gr.Button(value="\U0001F3B2")
                    seed = gr.Number(value=0, label="随机推理种子")
                with gr.Row():
                    prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz')
                    prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='录制prompt音频文件')
                prompt_text_cv = gr.Textbox(label="输入prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
                # instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='')
                seed_button.click(generate_seed, inputs=[], outputs=seed)
                mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
            generate_button = gr.Button("生成音频")
            audio_output = gr.Audio(label="合成音频")
            
            with gr.Column(variant='panel'):
                batch_size = gr.Slider(minimum=1, maximum=10, value=2, step=1, label='Talker Batch size')
    if not talk:
        character = gr.Radio(['女性角色', '男性角色', '自定义角色'], label="角色选择", value='自定义角色')
        character.change(fn=character_change, inputs=[character], outputs=[source_image])
        talker_method = gr.Radio(choices=['SadTalker', 'Wav2Lip', 'Wav2Lipv2', 'NeRFTalk', 'Comming Soon!!!'], value='SadTalker', label='数字人模型选择')
        talker_method.change(fn=talker_model_change, inputs=[talker_method], outputs=[talker_method])
    else:
        character = None
        talker_method = None
    tts_method = gr.Radio(['Edge-TTS', 'PaddleTTS', 'GPT-SoVITS克隆声音', 'CosyVoice-SFT模式', 'CosyVoice-克隆翻译模式', 'Comming Soon!!!'], label="Text To Speech Method", value='Edge-TTS')
    tts_method.change(fn=tts_model_change, inputs=[tts_method], outputs=[tts_method])
    asr_method = gr.Radio(choices=['Whisper-tiny', 'Whisper-base', 'FunASR', 'OmniSenseVoice-quantize', 'OmniSenseVoice', 'Comming Soon!!!'], value='Whisper-base', label='语音识别模型选择')
    asr_method.change(fn=asr_model_change, inputs=[asr_method], outputs=[asr_method])
    llm_method = gr.Dropdown(choices=['Qwen', 'Qwen2', 'Linly', 'Gemini', 'ChatGLM', 'ChatGPT', 'GPT4Free', 'QAnything', '直接回复 Direct Reply', 'Comming Soon!!!'], value='直接回复 Direct Reply', label='LLM 模型选择')
    llm_method.change(fn=llm_model_change, inputs=[llm_method], outputs=[llm_method])
    return (source_image, voice, rate, volume, pitch, am, voc, lang, male, 
            ref_audio, prompt_text, prompt_language, text_language, cut_method, use_mic_voice, tts_method, 
            batch_size, character, talker_method, asr_method, llm_method, generate_button, audio_output, 
            mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor)

def exmaple_setting(asr, text, character, talk, tts, voice, llm):
    # 默认text的Example
    examples = [
        ['Whisper-base', '应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-XiaoxiaoNeural', '直接回复 Direct Reply'],
        ['Whisper-tiny', '应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'PaddleTTS', 'None', '直接回复 Direct Reply'],
        ['Whisper-base', '应对压力最有效的方法是什么？', '女性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-XiaoxiaoNeural', 'Qwen'],
        ['FunASR', '如何进行时间管理？', '男性角色', 'SadTalker', 'Edge-TTS', 'zh-CN-YunyangNeural', 'Qwen'],
        ['Whisper-tiny', '为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？', '女性角色', 'Wav2Lip', 'PaddleTTS', 'None', 'Qwen'],
        ['Whisper-tiny', '为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？', '女性角色', 'Wav2Lipv2', 'Edge-TTS', 'None', 'Qwen'],
    ]
    with gr.Row(variant='panel'):
        with gr.Column(variant='panel'):
            gr.Markdown("## Test Examples")
            gr.Examples(
                examples = examples,
                inputs = [asr, text, character, talk , tts, voice, llm],
            )
def app_multi():
    with gr.Blocks(analytics_enabled=False, title='Linly-Talker') as inference:
        # 显示标题
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 多轮GPT对话"))
        
        with gr.Row():
            with gr.Column():
                # 加载 Web UI 设置
                (source_image, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 ref_audio, prompt_text, prompt_language, text_language, cut_method, use_mic_voice,
                 tts_method, batch_size, character, talker_method, asr_method, llm_method, generate_button, audio_output,
                 mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor) = webui_setting()

                
                # 数字人问答视频显示
                video = gr.Video(label='数字人问答', scale=0.5)
                video_button = gr.Button("🎬 生成数字人视频（对话后）", variant='primary')
            
            with gr.Column():
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('SadTalker数字人参数设置'):
                        with gr.Accordion("Advanced Settings", open=False):
                            gr.Markdown("SadTalker: need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more details")
                            with gr.Column(variant='panel'):
                                # 数字人参数设置
                                with gr.Row():
                                    pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0)
                                    exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1)
                                    blink_every = gr.Checkbox(label="use eye blink", value=True)
                                with gr.Row():
                                    size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model? 256 is faster")
                                    preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                                with gr.Row():
                                    is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                                    facerender = gr.Radio(['facevid2vid'], value='facevid2vid', label='facerender', info="which face render?")
                                with gr.Row():
                                    fps = gr.Slider(label='fps in generation', step=1, maximum=30, value=20)
                                    enhancer = gr.Checkbox(label="GFPGAN as Face enhancer(slow)")
                
                # System 设定及清除历史对话
                with gr.Row():
                    with gr.Column(scale=3):
                        system_input = gr.Textbox(value=DEFAULT_SYSTEM, lines=1, label='System (设定角色)')
                    with gr.Column(scale=1):
                        modify_system = gr.Button("🛠️ 设置system并清除历史对话", scale=2)
                    system_state = gr.Textbox(value=DEFAULT_SYSTEM, visible=False)
                
                # 聊天机器人界面
                chatbot = gr.Chatbot(height=400, show_copy_button=True)
                
                # 语音输入及识别按钮
                with gr.Group():
                    question_audio = gr.Audio(sources=['microphone','upload'], type="filepath", label='语音对话', autoplay=False)
                    asr_btn = gr.Button('🎤 语音识别（语音对话后点击）')
                
                # 文本输入框
                msg = gr.Textbox(label="输入文字/问题", lines=3, placeholder='请输入文本或问题，同时可以设置LLM模型。默认使用直接回复。')
                asr_btn.click(fn=Asr, inputs=[question_audio], outputs=[msg])
                

                generate_button.click(fn=TTS_response, 
                                      inputs=[msg, voice, rate, volume, pitch, am, voc, lang, male,
                                                ref_audio, prompt_text, prompt_language, text_language,
                                                cut_method, question_audio, prompt_text, use_mic_voice, 
                                                mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, tts_method, ],
                                      outputs=[audio_output])

                # 清除历史记录和提交按钮
                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话")
                    sumbit = gr.Button("🚀 发送", variant='primary')
                    
                # 设置按钮的点击事件
                sumbit.click(chat_response, inputs=[system_input, msg, chatbot], outputs=[msg, chatbot])
                clear_history.click(fn=clear_session, outputs=[msg, chatbot])
                modify_system.click(fn=modify_system_session, inputs=[system_input], outputs=[system_state, system_input, chatbot])
                video_button.click(fn=human_response, inputs=[source_image, chatbot, question_audio, talker_method, voice, rate, volume, pitch,
                                                             am, voc, lang, male, 
                                                             ref_audio, prompt_text, prompt_language, text_language, cut_method,  use_mic_voice, 
                                                             mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor,
                                                             tts_method, character, preprocess_type, 
                                                             is_still_mode, enhancer, batch_size, size_of_image,
                                                             pose_style, facerender, exp_weight, blink_every, fps], outputs=[video])

        # 示例设置
        exmaple_setting(asr_method, msg, character, talker_method, tts_method, voice, llm_method)
    return inference

def app_img():
    with gr.Blocks(analytics_enabled=False, title='Linly-Talker') as inference:
        # 显示标题
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) 个性化角色互动"))
        
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                # 加载 Web UI 设置
                (source_image, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 ref_audio, prompt_text, prompt_language, text_language, cut_method, use_mic_voice,
                 tts_method, batch_size, character, talker_method, asr_method, llm_method, generate_button, audio_output,
                 mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor) = webui_setting()
            
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem('对话'):
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone', 'upload'], type="filepath", label='语音对话')
                            input_text = gr.Textbox(label="输入文字/问题", lines=3, placeholder='请输入文本或问题，同时可以设置LLM模型。默认使用直接回复。')
                            asr_btn = gr.Button('语音识别（语音对话后点击）')
                        asr_btn.click(fn=Asr, inputs=[question_audio], outputs=[input_text])
                generate_button.click(fn=TTS_response, 
                                      inputs=[input_text, voice, rate, volume, pitch, am, voc, lang, male,
                                                ref_audio, prompt_text, prompt_language, text_language,
                                                cut_method, question_audio, prompt_text, use_mic_voice, 
                                                mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, tts_method, ],
                                      outputs=[audio_output])
                with gr.Tabs(elem_id="text_examples"): 
                    gr.Markdown("## Text Examples")
                    examples = [
                        ['应对压力最有效的方法是什么？'],
                        ['如何进行时间管理？'],
                        ['为什么有些人选择使用纸质地图或寻求方向，而不是依赖GPS设备或智能手机应用程序？'],
                    ]
                    gr.Examples(examples=examples, inputs=[input_text])
                
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('SadTalker数字人参数设置'):
                        with gr.Accordion("Advanced Settings", open=False):
                            gr.Markdown("SadTalker: need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more details")
                            with gr.Column(variant='panel'):
                                with gr.Row():
                                    pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style", value=0)
                                    exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1, label="expression scale", value=1)
                                    blink_every = gr.Checkbox(label="use eye blink", value=True)
                                with gr.Row():
                                    size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model? 256 is faster")
                                    preprocess_type = gr.Radio(['crop', 'resize', 'full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")
                                with gr.Row():
                                    is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                                    facerender = gr.Radio(['facevid2vid'], value='facevid2vid', label='facerender', info="which face render?")
                                with gr.Row():
                                    fps = gr.Slider(label='fps in generation', step=1, maximum=30, value=20)
                                    enhancer = gr.Checkbox(label="GFPGAN as Face enhancer(slow)")
                
                with gr.Tabs(elem_id="sadtalker_genearted"):
                    gen_video = gr.Video(label="数字人视频", format="mp4")

                submit = gr.Button('🎬 生成数字人视频', elem_id="sadtalker_generate", variant='primary')
                submit.click(
                    fn=Talker_response_img,
                    inputs=[question_audio, talker_method, input_text, voice, rate, volume, pitch,
                            am, voc, lang, male, ref_audio, prompt_text, prompt_language, text_language, cut_method, use_mic_voice,
                            mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, 
                            tts_method, source_image, preprocess_type, is_still_mode, enhancer, batch_size, size_of_image,
                            pose_style, facerender, exp_weight, blink_every, fps], 
                    outputs=[gen_video]
                )
        
        with gr.Row():
            examples = [
                ['examples/source_image/full_body_2.png', 'SadTalker', 'crop', False, False],
                ['examples/source_image/full_body_1.png', 'Wav2Lipv2', 'full', False, False],
                ['examples/source_image/full_body_2.png', 'Wav2Lipv2', 'full', False, False],
                ['examples/source_image/full_body_1.png', 'Wav2Lip', 'full', True, False],
                ['examples/source_image/full_body_1.png', 'SadTalker', 'full', True, False],
                ['examples/source_image/full4.jpeg', 'SadTalker', 'crop', False, True],
            ]
            gr.Examples(
                examples=examples,
                inputs=[source_image, talker_method, preprocess_type, is_still_mode, enhancer],
                outputs=[gen_video],
                # cache_examples=True,
            )
    return inference


def load_musetalk_model():
    """加载MuseTalk模型，显示加载状态和结果信息。"""
    gr.Warning("若显存不足，可能会导致模型加载失败，可以尝试使用其他模型或者换其他设备。")
    gr.Info("MuseTalk模型导入中...")
    musetalker.init_model()
    gr.Info("MuseTalk模型导入成功")
    return "MuseTalk模型导入成功"

def musetalk_prepare_material(source_video, bbox_shift):
    """准备MuseTalk所需的素材，检查模型是否已加载。"""
    if musetalker.load is False:
        gr.Warning("请先加载MuseTalk模型后重新上传文件")
        return source_video, None
    return musetalker.prepare_material(source_video, bbox_shift)

def app_muse():
    """定义MuseTalk应用的UI和逻辑。"""
    with gr.Blocks(analytics_enabled=False, title='Linly-Talker') as inference:
        gr.HTML(get_title("Linly 智能对话系统 (Linly-Talker) MuseTalker数字人实时对话"))

        # 上传参考视频和调整bbox_shift
        with gr.Row(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.TabItem('MuseV Video'):
                    gr.Markdown("MuseV: 需要帮助？请访问 [MuseVDemo](https://huggingface.co/spaces/AnchorFake/MuseVDemo) 生成视频。")
                    source_video = gr.Video(label="Reference Video", sources=['upload'])
                    gr.Markdown(
                        "BBox_shift 推荐值下限，在生成初始结果后生成相应的 bbox 范围。"
                        "一般来说，正值（向下半部分移动）通常会增加嘴巴的张开度，"
                        "而负值（向上半部分移动）通常会减少嘴巴的张开度。"
                        "用户可根据具体需求调整此参数。"
                    )
                    bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
                    bbox_shift_scale = gr.Textbox(label="bbox_shift_scale", value="", interactive=False)
                
                # 加载MuseTalk模型按钮
                load_musetalk = gr.Button("加载MuseTalk模型(传入视频前先加载)", variant='primary')
                load_musetalk.click(fn=load_musetalk_model, outputs=bbox_shift_scale)

                # 加载 Web UI 设置
                (_, voice, rate, volume, pitch, 
                 am, voc, lang, male, 
                 ref_audio, prompt_text, prompt_language, text_language, cut_method, use_mic_voice,
                 tts_method, batch_size, _, _, asr_method, llm_method, generate_button, audio_output,
                 mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor) = webui_setting(talk=True)
            
            # 处理source_video变化
            source_video.change(fn=musetalk_prepare_material, inputs=[source_video, bbox_shift], outputs=[source_video, bbox_shift_scale])

            # 问题输入和ASR识别
            with gr.Column(variant='panel'):
                with gr.Tabs():
                    with gr.TabItem('对话'):
                        with gr.Group():
                            question_audio = gr.Audio(sources=['microphone', 'upload'], type="filepath", label='语音对话')
                            input_text = gr.Textbox(label="输入文字/问题", lines=3, placeholder='请输入文本或问题，同时可以设置LLM模型。默认使用直接回复。')
                            asr_btn = gr.Button('语音识别（语音对话后点击）')
                        asr_btn.click(fn=Asr, inputs=[question_audio], outputs=[input_text]) 
                    generate_button.click(fn=TTS_response, 
                                      inputs=[input_text, voice, rate, volume, pitch, am, voc, lang, male,
                                                ref_audio, prompt_text, prompt_language, text_language,
                                                cut_method, question_audio, prompt_text, use_mic_voice, 
                                                mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, tts_method, ],
                                      outputs=[audio_output])

                # 生成MuseTalk视频
                with gr.TabItem("MuseTalk Video"):
                    gen_video = gr.Video(label="数字人视频", format="mp4")
                submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                # examples = [os.path.join('Musetalk/data/video', video) for video in os.listdir("Musetalk/data/video")]
                
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
                    inputs=[source_video, bbox_shift], 
                )

            # 提交按钮点击事件
            submit.click(
                fn=MuseTalker_response,
                inputs=[
                    source_video, bbox_shift, question_audio, input_text, 
                    voice, rate, volume, pitch, am, voc, lang, male, 
                    ref_audio, prompt_text, prompt_language, text_language, cut_method, use_mic_voice, 
                    mode_checkbox_group, sft_dropdown, prompt_text_cv, prompt_wav_upload, prompt_wav_record, seed, speed_factor, 
                    tts_method, batch_size
                ],
                outputs=[gen_video]
            )

    return inference
def asr_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    """根据选择的模型名称更换ASR模型。"""
    global asr
    clear_memory()  # 清理显存

    try:
        if model_name == "Whisper-tiny":
            asr_path = 'Whisper/tiny.pt' if os.path.exists('Whisper/tiny.pt') else 'tiny'
            asr = WhisperASR(asr_path)
            gr.Info("Whisper-tiny模型导入成功")
        elif model_name == "Whisper-base":
            asr_path = 'Whisper/base.pt' if os.path.exists('Whisper/base.pt') else 'base'
            asr = WhisperASR(asr_path)
            gr.Info("Whisper-base模型导入成功")
        elif model_name == 'FunASR':
            from ASR import FunASR
            asr = FunASR()
            gr.Info("FunASR模型导入成功")
        elif model_name == 'OmniSenseVoice-quantize':
            from ASR import OmniSenseVoice
            asr = OmniSenseVoice(quantize=True)
            gr.Info("OmniSenseVoice-quantize模型导入成功")
        elif model_name == 'OmniSenseVoice':
            from ASR import OmniSenseVoice
            asr = OmniSenseVoice(quantize=False)
            gr.Info("OmniSenseVoice模型导入成功")
        else:
            gr.Warning("未知ASR模型，可提issue和PR 或者 建议更新模型")
    except Exception as e:
        gr.Warning(f"{model_name}模型加载失败: {e}")

    return model_name

def llm_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    """更换LLM模型，并根据选择的模型加载相应资源。"""
    global llm
    gemini_apikey = ""  # Gemini模型的API密钥
    openai_apikey = ""  # OpenAI的API密钥
    proxy_url = None  # 代理URL

    # 清理显存，释放不必要的显存以便加载新模型
    clear_memory()

    try:
        if model_name == 'Linly':
            llm = llm_class.init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf', prefix_prompt=PREFIX_PROMPT)
            gr.Info("Linly模型导入成功")
        elif model_name == 'Qwen':
            llm = llm_class.init_model('Qwen', 'Qwen/Qwen-1_8B-Chat', prefix_prompt=PREFIX_PROMPT)
            gr.Info("Qwen模型导入成功")
        elif model_name == 'Qwen2':
            llm = llm_class.init_model('Qwen2', 'Qwen/Qwen1.5-0.5B-Chat', prefix_prompt=PREFIX_PROMPT)
            gr.Info("Qwen2模型导入成功")
        elif model_name == 'Gemini':
            if gemini_apikey:
                llm = llm_class.init_model('Gemini', 'gemini-pro', gemini_apikey, proxy_url)
                gr.Info("Gemini模型导入成功")
            else:
                gr.Warning("请填写Gemini的API密钥")
        elif model_name == 'ChatGLM':
            llm = llm_class.init_model('ChatGLM', 'THUDM/chatglm3-6b', prefix_prompt=PREFIX_PROMPT)
            gr.Info("ChatGLM模型导入成功")
        elif model_name == 'ChatGPT':
            if openai_apikey:
                llm = llm_class.init_model('ChatGPT', api_key=openai_apikey, proxy_url=proxy_url, prefix_prompt=PREFIX_PROMPT)
                gr.Info("ChatGPT模型导入成功")
            else:
                gr.Warning("请填写OpenAI的API密钥")
        elif model_name == '直接回复 Direct Reply':
            llm = llm_class.init_model(model_name)
            gr.Info("直接回复，不使用LLM模型")
        elif model_name == 'GPT4Free':
            llm = llm_class.init_model('GPT4Free', prefix_prompt=PREFIX_PROMPT)
            gr.Info("GPT4Free模型导入成功，请注意该模型可能不稳定")
        elif model_name == 'QAnything':
            llm = llm_class.init_model('QAnything')
            gr.Info("QAnything模型接口加载成功")
        else:
            gr.Warning("未知LLM模型，请检查模型名称或提出Issue")
    except Exception as e:
        gr.Warning(f"{model_name}模型加载失败: {e}")

    return model_name
def talker_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    """更换数字人对话模型，并根据选择的模型加载相应资源。"""
    global talker

    # 清理显存，释放不必要的显存以便加载新模型
    clear_memory()

    if model_name not in ['SadTalker', 'Wav2Lip', 'Wav2Lipv2', 'NeRFTalk']:
        gr.Warning("其他模型暂未集成，请等待更新")
        return model_name

    try:
        if model_name == 'SadTalker':
            from TFG import SadTalker
            talker = SadTalker(lazy_load=True)
            gr.Info("SadTalker模型导入成功")
        elif model_name == 'Wav2Lip':
            from TFG import Wav2Lip
            clear_memory()
            talker = Wav2Lip("checkpoints/wav2lip_gan.pth")
            gr.Info("Wav2Lip模型导入成功")
        elif model_name == 'Wav2Lipv2':
            from TFG import Wav2Lipv2
            clear_memory()
            talker = Wav2Lipv2('checkpoints/wav2lipv2.pth')
            gr.Info("Wav2Lipv2模型导入成功，能够生成更高质量的结果")
        elif model_name == 'NeRFTalk':
            from TFG import NeRFTalk
            talker = NeRFTalk()
            talker.init_model('checkpoints/Obama_ave.pth', 'checkpoints/Obama.json')
            gr.Info("NeRFTalk模型导入成功")
            gr.Warning("NeRFTalk模型仅针对单个人训练，内置奥巴马模型，上传其他图片无效")
    except Exception as e:
        gr.Warning(f"{model_name}模型加载失败: {e}")

    return model_name

def tts_model_change(model_name, progress=gr.Progress(track_tqdm=True)):
    """更换TTS模型，并根据选择的模型加载相应资源。"""
    global tts
    global cosyvoice
    # 清理显存，释放不必要的显存以便加载新模型
    clear_memory()

    try:
        if model_name == 'Edge-TTS':
            # tts = EdgeTTS()  # Uncomment when implementation available
            if edgetts.network:
                gr.Info("EdgeTTS模型导入成功")
            else:
                gr.Warning("EdgeTTS模型加载失败，请检查网络连接")
        elif model_name == 'PaddleTTS':
            from TTS import PaddleTTS
            tts = PaddleTTS()
            gr.Info("PaddleTTS模型导入成功, 效果有限，不建议使用")
        elif model_name == 'GPT-SoVITS克隆声音':
            gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
            vits.load_model(gpt_path, sovits_path)
            gr.Info("GPT-SoVITS模型加载成功，请上传参考音频进行克隆")
        elif model_name == 'CosyVoice-SFT模式':
            from VITS import CosyVoiceTTS
            model_path = 'checkpoints/CosyVoice_ckpt/CosyVoice-300M-SFT'
            cosyvoice = CosyVoiceTTS(model_path)
            gr.Info("CosyVoice模型导入成功，适合使用SFT模式，用微调后数据")
        elif model_name == 'CosyVoice-克隆翻译模式':
            from VITS import CosyVoiceTTS
            model_path = 'checkpoints/CosyVoice_ckpt/CosyVoice-300M'
            cosyvoice = CosyVoiceTTS(model_path)
            gr.Info("CosyVoice模型导入成功，更适合进行克隆声音和翻译声音")
        else:
            gr.Warning("未知TTS模型，请检查模型名称或提出Issue")
    except Exception as e:
        gr.Warning(f"{model_name}模型加载失败: {e}")

    return model_name

def success_print(text):
    """输出绿色文本，表示成功信息。"""
    print(f"\033[1;32m{text}\033[0m")

def error_print(text):
    """输出红色文本，表示错误信息。"""
    print(f"\033[1;31m{text}\033[0m")
    
if __name__ == "__main__":
    # 初始化LLM类
    llm_class = LLM(mode='offline')
    llm = llm_class.init_model('直接回复 Direct Reply')
    success_print("默认不使用LLM模型，直接回复问题，同时减少显存占用！")

    # 尝试加载GPT-SoVITS模块
    try:
        from VITS import *
        vits = GPT_SoVITS()
        success_print("Success! GPT-SoVITS模块加载成功，语音克隆默认使用GPT-SoVITS模型")
    except Exception as e:
        error_print(f"GPT-SoVITS 加载失败: {e}")
        error_print("如果使用VITS，请先下载GPT-SoVITS模型并安装环境")

    # 尝试加载SadTalker模块
    try:
        from TFG import SadTalker
        talker = SadTalker(lazy_load=True)
        success_print("Success! SadTalker模块加载成功，默认使用SadTalker模型")
    except Exception as e:
        error_print(f"SadTalker 加载失败: {e}")
        error_print("如果使用SadTalker，请先下载SadTalker模型")

    # 尝试加载Whisper ASR模块
    try:
        from ASR import WhisperASR
        asr = WhisperASR('base')
        success_print("Success! WhisperASR模块加载成功，默认使用Whisper-base模型")
    except Exception as e:
        error_print(f"WhisperASR 加载失败: {e}")
        error_print("如果使用FunASR，请先下载WhisperASR模型并安装环境")

    # 检查GPU显存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        if gpu_memory < 8:
            error_print("警告: 您的显卡显存小于8GB，不建议使用MuseTalk功能")

    # 尝试加载MuseTalk模块
    try:
        from TFG import MuseTalk_RealTime
        musetalker = MuseTalk_RealTime()
        success_print("Success! MuseTalk模块加载成功")
    except Exception as e:
        error_print(f"MuseTalk 加载失败: {e}")
        error_print("如果使用MuseTalk，请先下载MuseTalk模型")

    # 尝试加载EdgeTTS模块
    try:
        tts = edgetts
        if not tts.network:
            error_print("EdgeTTS模块加载失败，请检查网络连接")
    except Exception as e:
        error_print(f"EdgeTTS 加载失败: {e}")

    # Gradio UI的初始化和启动
    gr.close_all()
    demo_img = app_img()
    demo_multi = app_multi()
    demo_muse = app_muse()
    demo = gr.TabbedInterface(
        interface_list=[demo_img, demo_multi, demo_muse],
        tab_names=["个性化角色互动", "数字人多轮智能对话", "MuseTalk数字人实时对话"],
        title="Linly-Talker WebUI"
    )
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(
        server_name=ip,  # 本地localhost:127.0.0.1 或 "0.0.0.0" 进行全局端口转发
        server_port=port,
        # ssl_certfile=ssl_certfile,  # SSL证书文件
        # ssl_keyfile=ssl_keyfile,  # SSL密钥文件
        # ssl_verify=False,
        # share=True,
        debug=True,
    )