import os
from paddlespeech.cli.tts.infer import TTSExecutor

"""
PaddleSpeech

声码器说明：这里预制了三种声码器【PWGan】【WaveRnn】【HifiGan】, 三种声码器效果和生成时间有比较大的差距，请跟进自己的需要进行选择。不过只选择了前两种，因为WaveRNN太慢了

| 声码器 | 音频质量 | 生成速度 |
| :----: | :----: | :----: |
| PWGan | 中等 | 中等 |
| WaveRnn | 高 | 非常慢（耐心等待） |
| HifiGan | 低 | 快 |

这些PaddleSpeech中的样例主要按数据集分类，我们主要使用的TTS数据集有：

CSMCS (普通话单发音人)
AISHELL3 (普通话多发音人)
LJSpeech (英文单发音人)
VCTK (英文多发音人)

PaddleSpeech 的 TTS 模型具有以下映射关系：

tts0 - Tacotron2
tts1 - TransformerTTS
tts2 - SpeedySpeech
tts3 - FastSpeech2
voc0 - WaveFlow
voc1 - Parallel WaveGAN
voc2 - MelGAN
voc3 - MultiBand MelGAN
voc4 - Style MelGAN
voc5 - HiFiGAN
vc0 - Tacotron2 Voice Clone with GE2E
vc1 - FastSpeech2 Voice Clone with GE2E

以下是 PaddleSpeech 提供的可以被命令行和 python API 使用的预训练模型列表：

- 声学模型
  | 模型 | 语言 |
  | :--- | :---: |
  |      speedyspeech_csmsc      |    zh    |
  |      fastspeech2_csmsc       |    zh    |
  |     fastspeech2_ljspeech     |    en    |
  |     fastspeech2_aishell3     |    zh    |
  |       fastspeech2_vctk       |    en    |
  | fastspeech2_cnndecoder_csmsc |    zh    |
  |       fastspeech2_mix        |   mix    |
  |       tacotron2_csmsc        |    zh    |
  |      tacotron2_ljspeech      |    en    |
  |       fastspeech2_male       |    zh    |
  |       fastspeech2_male       |    en    |
  |       fastspeech2_male       |   mix    |
  |       fastspeech2_canton     |  canton  |

- 声码器
  | 模型 | 语言 |
  | :--- | :---: |
  |         pwgan_csmsc          |    zh    |
  |        pwgan_ljspeech        |    en    |
  |        pwgan_aishell3        |    zh    |
  |          pwgan_vctk          |    en    |
  |       mb_melgan_csmsc        |    zh    |
  |      style_melgan_csmsc      |    zh    |
  |        hifigan_csmsc         |    zh    |
  |       hifigan_ljspeech       |    en    |
  |       hifigan_aishell3       |    zh    |
  |         hifigan_vctk         |    en    |
  |        wavernn_csmsc         |    zh    |
  |         pwgan_male           |    zh    |
  |        hifigan_male          |    zh    |
"""


class PaddleTTS:
    def __init__(self) -> None:
        pass
        
    def predict(self, text, am, voc, spk_id = 174, lang = 'zh', male=False, save_path = 'output.wav'):
        self.tts = TTSExecutor()
        
        use_onnx = True
        voc = voc.lower()
        am = am.lower()
        
        if male:
            assert voc in ["pwgan", "hifigan"], "male voc must be 'pwgan' or 'hifigan'"
            wav_file = self.tts(
            text = text,
            output = save_path,
            am='fastspeech2_male',
            voc= voc + '_male',
            lang=lang,
            use_onnx=use_onnx
            )
            return wav_file
    
        assert am in ['tacotron2', 'fastspeech2'], "am must be 'tacotron2' or 'fastspeech2'"
        
        # 混合中文英文语音合成
        if lang == 'mix':
            # mix只有fastspeech2
            am = 'fastspeech2_mix'
            voc += '_csmsc'
        # 英文语音合成
        elif lang == 'en':
            am += '_ljspeech'
            voc += '_ljspeech'
        # 中文语音合成
        elif lang == 'zh':
            assert voc in ['wavernn', 'pwgan', 'hifigan', 'style_melgan', 'mb_melgan'], "voc must be 'wavernn' or 'pwgan' or 'hifigan' or 'style_melgan' or 'mb_melgan'"
            am += '_csmsc'
            voc += '_csmsc'
        elif lang == 'canton':
            am = 'fastspeech2_canton'
            voc = 'pwgan_aishell3'
            spk_id = 10
        print("am:", am, "voc:", voc, "lang:", lang, "male:", male, "spk_id:", spk_id)
        try:
            cmd = f'paddlespeech tts --am {am} --voc {voc} --input "{text}" --output {save_path} --lang {lang} --spk_id {spk_id} --use_onnx {use_onnx}'
            os.system(cmd)
            wav_file = save_path
        except:
            # 语音合成
            wav_file = self.tts(
                text = text,
                output = save_path,
                am = am,
                voc = voc,
                lang = lang,
                spk_id = spk_id,
                use_onnx=use_onnx
                )
        return wav_file 
        
if __name__ == "__main__":
    tts = PaddleTTS()
    tts.predict("Hello world", 'FastSpeech2', 'PWGan', spk_id=174, lang='en', male=False, save_path='output.wav')