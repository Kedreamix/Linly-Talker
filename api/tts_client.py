import os
import requests

tts_service_host = os.environ.get("TTS_SERVICE_HOST", "localhost")
tts_service_port = os.environ.get("TTS_SERVICE_PORT", 8001)

# API endpoint URLs
CHANGE_MODEL_URL = f"http://{tts_service_host}:{tts_service_port}/tts_change_model/"
TTS_RESPONSE_URL = f"http://{tts_service_host}:{tts_service_port}/tts_response/"

def change_model(model_name):
    """请求更换TTS模型"""
    response = requests.post(CHANGE_MODEL_URL, params={"model_name": model_name})
    if response.status_code == 200:
        print(f"模型更换成功: {response.json()}")
    else:
        print(f"模型更换失败: {response.status_code}, {response.text}")

def request_tts(payload, ref_audio_path=None, output_wav_path='output_tts.wav'):
    """请求TTS生成音频，支持上传文件"""
    files = {}
    if ref_audio_path:
        files['ref_audio'] = open(ref_audio_path, 'rb')
    try:
        response = requests.post(TTS_RESPONSE_URL, data=payload, files=files)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(output_wav_path, 'wb') as wav_file:
            wav_file.write(response.content)    
        print(f"TTS生成成功, 音频保存为: {output_wav_path}")
    except requests.RequestException as e:
        print(f"TTS生成失败: {e}")

if __name__ == "__main__":
    result_dir = "outputs"
    os.makedirs(result_dir, exist_ok=True)

    # 要测试的模型列表
    models = [
        "EdgeTTS",
        "PaddleTTS",
    ]

    # 循环更换模型并生成TTS
    for model_name in models:
        print(f"切换到模型: {model_name}")
        change_model(model_name)

        # 请求TTS生成音频
        text = "你好，我是Linly-Talker，很高兴认识大家"
        payload = {
            "text": text,
            "tts_method": model_name,
            "save_path": f"output_{model_name}.wav"
        }
        output_wav_path = os.path.join(result_dir, f"output_{model_name}.wav")
        request_tts(payload, output_wav_path=output_wav_path)
        print("\n" + "-" * 50 + "\n")

    # 测试 GPT-SoVITS
    model_name = "GPT-SoVITS克隆声音"
    print(f"切换到模型: {model_name}")
    change_model(model_name)

    # 请求TTS生成音频
    payload = {
        "text": "你好，我是Linly-Talker，很高兴认识大家",
        "tts_method": model_name,
        "prompt_text": "你好，我是Linly-Talker，我是克隆生成的",
        "ref_text": "你好，我是Linly-Talker。",
        "prompt_language": "中文",
        "ref_language": "中文",
        "save_path": f"output_{model_name}.wav"
    }
    ref_audio_path = os.path.join(result_dir, "output_EdgeTTS.wav")
    output_wav_path = os.path.join(result_dir, "output_GPT_SoVITS.wav")
    request_tts(payload, ref_audio_path=ref_audio_path, output_wav_path=output_wav_path)
    print("\n" + "-" * 50 + "\n")

    # 测试 CosyVoice
    cosyvoice_models = [
        "CosyVoice-SFT模式",
        "CosyVoice-克隆翻译模式"
    ]

    for cosy_model in cosyvoice_models:
        print(f"切换到模型: {cosy_model}")
        change_model(cosy_model)
        # 请求TTS生成音频
        payload = {
            "text": "你们好，今天天气很好，你们都要天天开心哦",
            "tts_method": cosy_model,
            "cosyvoice_mode": "预训练音色" if "SFT模式" in cosy_model else "3s极速复刻",
            "ref_text": "你好，我是Linly-Talker，很高兴认识大家",
        }
        ref_audio_path = os.path.join(result_dir, "output_EdgeTTS.wav") if "克隆翻译模式" in cosy_model else None
        output_wav_path = os.path.join(result_dir, f"output_{cosy_model}.wav")
        request_tts(payload, ref_audio_path=ref_audio_path, output_wav_path=output_wav_path)
        print("\n" + "-" * 50 + "\n")