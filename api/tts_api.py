from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, sys
import gc, torch
from loguru import logger
import asyncio
from typing import Optional

sys.path.append("./")

app = FastAPI()

tts = None
cosyvoice = None
edgetts = None
vits = None

class TTSRequest(BaseModel):
    text: str = '你好，我是Linly-Talker。'
    voice: str = 'zh-CN-XiaoxiaoNeural'
    rate: float = 1.0
    volume: float = 1.0
    pitch: float = 1.0
    speed_factor: float = 1.0
    am: str = 'FastSpeech2'
    voc: str = 'PWGan'
    lang: str = 'zh'
    male: bool = False
    prompt_text: str = ''
    prompt_language: str = '中文'
    ref_audio: str = ''
    ref_text: str = ''
    ref_language: str = '中文'
    cut_method: str = '凑四句一切'
    cosyvoice_mode: str = '预训练音色'
    sft_dropdown: str = '中文男'
    seed: int = 42
    tts_method: str = 'EdgeTTS'
    save_path: str = 'answer.wav'

@app.post("/tts_change_model/")
async def change_model(model_name: str = Query(..., description="要加载的TTS模型名称")):
    global tts, cosyvoice, edgetts, vits

    await clear_memory()

    try:
        if model_name == 'EdgeTTS':
            from TTS import EdgeTTS
            if edgetts is None:
                edgetts = EdgeTTS()
            if edgetts.network:
                logger.info("EdgeTTS模型加载成功")
            else:
                logger.warning("EdgeTTS模型加载失败，请检查网络连接")
                raise HTTPException(status_code=503, detail="EdgeTTS模型加载失败，请检查网络连接")
        elif model_name == 'PaddleTTS':
            from TTS import PaddleTTS
            if tts is None:
                tts = PaddleTTS()
            logger.info("PaddleTTS模型加载成功")
        elif model_name == 'GPT-SoVITS克隆声音':
            gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
            if vits is None:
                from VITS import GPT_SoVITS
                vits = GPT_SoVITS()
            vits.load_model(gpt_path, sovits_path)
            logger.info("GPT-SoVITS模型加载成功")
        elif model_name == 'CosyVoice-SFT模式':
            from VITS import CosyVoiceTTS
            model_path = 'checkpoints/CosyVoice_ckpt/CosyVoice-300M-SFT'
            if cosyvoice is None:
                cosyvoice = CosyVoiceTTS(model_path)
            logger.info("CosyVoice-SFT模式模型加载成功")
        elif model_name == 'CosyVoice-克隆翻译模式':
            from VITS import CosyVoiceTTS
            model_path = 'checkpoints/CosyVoice_ckpt/CosyVoice-300M'
            if cosyvoice is None:
                cosyvoice = CosyVoiceTTS(model_path)
            logger.info("CosyVoice-克隆翻译模式模型加载成功")
        else:
            logger.warning(f"未知的TTS模型: {model_name}")
            raise HTTPException(status_code=400, detail=f"未知的TTS模型: {model_name}")
    except ImportError as e:
        logger.error(f"导入模型 {model_name} 失败: {e}")
        raise HTTPException(status_code=500, detail=f"导入模型 {model_name} 失败: {e}")
    except Exception as e:
        logger.error(f"{model_name} 模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"{model_name} 模型加载失败: {e}")

    return {"message": f"{model_name} 模型加载成功"}

async def clear_memory():
    logger.info("清理显存资源")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info(f"显存使用情况: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """保存上传的文件到指定路径"""
    with open(destination, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return destination

def predict_edge_tts(request: TTSRequest):
    global edgetts
    if edgetts is None:
        raise HTTPException(status_code=400, detail="EdgeTTS 模型未加载")
    if not edgetts.network:
        raise HTTPException(status_code=503, detail="EdgeTTS 模型网络问题")

    try:
        edgetts.predict(request.text, request.voice, request.rate, request.volume, request.pitch, request.save_path, 'answer.vtt')
    except Exception as e:
        os.system(f'edge-tts --text "{request.text}" --voice {request.voice} --write-media {request.save_path} --write-subtitles answer.vtt')

    return request.save_path

def predict_paddle_tts(request: TTSRequest):
    global tts
    if tts is None:
        raise HTTPException(status_code=400, detail="PaddleTTS 模型未加载")
    
    try:
        tts.predict(request.text, request.am, request.voc, lang=request.lang, male=request.male, save_path=request.save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PaddleTTS 预测失败: {e}")

    return request.save_path

def predict_gpt_sovits(request: TTSRequest):
    global vits
    if vits is None:
        raise HTTPException(status_code=400, detail="GPT-SoVITS 模型未加载")
    
    
    
    try:
        vits.predict(ref_wav_path=request.ref_audio, prompt_text=request.prompt_text,
                     prompt_language=request.prompt_language, text=request.text,
                     text_language=request.ref_language, how_to_cut=request.cut_method,
                     save_path=request.save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-SoVITS 预测失败: {e}")

    return request.save_path

def predict_cosyvoice(request: TTSRequest):
    global cosyvoice
    if cosyvoice is None:
        raise HTTPException(status_code=400, detail="CosyVoice 模型未加载")

    prompt_wav = None
    if request.ref_audio:
        prompt_wav = request.ref_audio

    if request.cosyvoice_mode in ['3s极速复刻', '跨语种复刻'] and not prompt_wav:
        raise HTTPException(status_code=400, detail="选择的模式需要提供 prompt 音频")

    try:
        if request.cosyvoice_mode == '预训练音色':
            output = cosyvoice.predict_sft(request.text, request.sft_dropdown, speed_factor=request.speed_factor, save_path=request.save_path)
        elif request.cosyvoice_mode == '3s极速复刻':
            output = cosyvoice.predict_zero_shot(request.text, request.ref_text, prompt_wav, speed_factor=request.speed_factor, save_path=request.save_path)
        elif request.cosyvoice_mode == '跨语种复刻':
            output = cosyvoice.predict_cross_lingual(request.text, prompt_wav, speed_factor=request.speed_factor, save_path=request.save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CosyVoice 预测失败: {e}")

    return output

@app.post("/tts_response/")
async def tts_response(
    text: str = Form('你好，我是Linly-Talker。'),
    voice: str = Form('zh-CN-XiaoxiaoNeural'),
    rate: float = Form(1.0),
    volume: float = Form(1.0),
    pitch: float = Form(1.0),
    speed_factor: float = Form(1.0),
    am: str = Form('FastSpeech2'),
    voc: str = Form('PWGan'),
    lang: str = Form('zh'),
    male: bool = Form(False),
    prompt_text: str = Form(''),
    prompt_language: str = Form('中文'),
    ref_text: str = Form(''),
    ref_language: str = Form('中文'),
    cut_method: str = Form('凑四句一切'),
    cosyvoice_mode: str = Form('预训练音色'),
    sft_dropdown: str = Form('中文男'),
    seed: int = Form(42),
    tts_method: str = Form('EdgeTTS'),
    save_path: str = Form('answer.wav'),
    ref_audio: Optional[UploadFile] = File(None)
):
    ref_audio_path = None
    if ref_audio:
        # 保存上传的音频文件
        ref_audio_path = save_upload_file(ref_audio, "uploaded_ref_audio.wav")

    request = TTSRequest(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        speed_factor=speed_factor,
        am=am,
        voc=voc,
        lang=lang,
        male=male,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        ref_audio=ref_audio_path if ref_audio else '',
        ref_text=ref_text,
        ref_language=ref_language,
        cut_method=cut_method,
        cosyvoice_mode=cosyvoice_mode,
        sft_dropdown=sft_dropdown,
        seed=seed,
        tts_method=tts_method,
        save_path=save_path
    )
    # print(request)
    
    if not request.text:
        raise HTTPException(status_code=400, detail="文本内容为空")

    try:
        if request.tts_method == 'EdgeTTS':
            file_path = predict_edge_tts(request)
        elif request.tts_method == 'PaddleTTS':
            file_path = predict_paddle_tts(request)
        elif request.tts_method == 'GPT-SoVITS克隆声音':
            file_path = predict_gpt_sovits(request)
        elif "CosyVoice" in request.tts_method:
            file_path = predict_cosyvoice(request)
        else:
            raise HTTPException(status_code=400, detail=f"未知的TTS方法: {request.tts_method}")
        
        if os.path.exists(request.save_path):
            return FileResponse(file_path, media_type='audio/wav', filename=request.save_path)
        else:
            logger.error(f"处理TTS请求失败: {e}")
            raise HTTPException(status_code=404, detail="Audio file not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理TTS请求失败: {e}")
    # finally:
    #     if ref_audio:
    #         os.remove(ref_audio_path)
    #     os.remove(request.save_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
