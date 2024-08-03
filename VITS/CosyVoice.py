import os, sys
sys.path.append('CosyVoice/third_party/Matcha-TTS')
sys.path.append('CosyVoice/')
import torch
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, speed_change
import librosa
import torchaudio

class CosyVoiceTTS:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = CosyVoice(model_path)
        
    # SFT usage
    def predict_sft(self, text, spks, save_path='sft.wav', speed_factor = 1.0):
        assert spks in self.model.list_avaliable_spks() and 'SFT' in self.model_path
        output = self.model.inference_sft(text, spks)
        if speed_factor != 1.0:
            output['tts_speech'] = self.speed_change(output['tts_speech'], speed = speed_factor)
        torchaudio.save(save_path, output['tts_speech'], 22050)
        return save_path

    def predict_zero_shot(self, text, prompt_text, prompt_speech, save_path='zero_shot.wav', speed_factor = 1.0):
        prompt_speech_16k = self.postprocess(load_wav(prompt_speech, 16000))
        output = self.model.inference_zero_shot(text, prompt_text, prompt_speech_16k)
        if speed_factor != 1.0:
            output['tts_speech'] = self.speed_change(output['tts_speech'], speed = speed_factor)
        torchaudio.save(save_path, output['tts_speech'], 22050)
        return save_path

    def predict_cross_lingual(self,prompt_text, prompt_speech, save_path='cross_lingual.wav', speed_factor = 1.0):
        prompt_speech_16k = self.postprocess(load_wav(prompt_speech, 16000))
        output = self.model.inference_cross_lingual(prompt_text, prompt_speech_16k)
        if speed_factor != 1.0:
            output['tts_speech'] = self.speed_change(output['tts_speech'], speed = speed_factor)
        torchaudio.save(save_path, output['tts_speech'], 22050)
        return save_path

    def speed_change(self, wav, target_sr = 22050, speed=1.0):
        return speed_change(wav, target_sr, str(speed))


    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440, target_sr=22050):
        max_val = 0.8
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
        return speech
    
if __name__ == "__main__":
    # SFT model example
    cosyvoice_sft = CosyVoiceTTS('checkpoints/CosyVoice_ckpt/CosyVoice-300M-SFT')
    print(cosyvoice_sft.model.list_avaliable_spks())
    cosyvoice_sft.predict_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', save_path='sft_output.wav')

    # Zero-shot model example
    cosyvoice_zero_shot = CosyVoiceTTS('checkpoints/CosyVoice_ckpt/CosyVoice-300M')
    prompt_speech = 'zero_shot_prompt.wav'
    cosyvoice_zero_shot.predict_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。',
                                          prompt_speech, save_path='zero_shot_output.wav')