import LangSegment
import numpy as np
import librosa
import torch
import re, os
import librosa
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
sys.path.append('GPT_SoVITS/')
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from feature_extractor import cnhubert
from my_utils import load_audio
from module.mel_processing import spectrogram_torch
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from scipy.io.wavfile import write
from time import time as ttime

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

is_half = True
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half=False

if device=="cpu":
    is_half=False

dtype=torch.float16 if is_half == True else torch.float32
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
cnhubert.cnhubert_base_path = cnhubert_base_path

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language.replace("all_",""))
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert

def splite_en_inf(sentence, language):
    pattern = re.compile(r'[a-zA-Z ]+')
    textlist = []
    langlist = []
    pos = 0
    for match in pattern.finditer(sentence):
        start, end = match.span()
        if start > pos:
            textlist.append(sentence[pos:start])
            langlist.append(language)
        textlist.append(sentence[start:end])
        langlist.append("en")
        pos = end
    if pos < len(sentence):
        textlist.append(sentence[pos:])
        langlist.append(language)
    # Merge punctuation into previous word
    for i in range(len(textlist)-1, 0, -1):
        if re.match(r'^[\W_]+$', textlist[i]):
            textlist[i-1] += textlist[i]
            del textlist[i]
            del langlist[i]
    # Merge consecutive words with the same language tag
    i = 0
    while i < len(langlist) - 1:
        if langlist[i] == langlist[i+1]:
            textlist[i] += textlist[i+1]
            del textlist[i+1]
            del langlist[i+1]
        else:
            i += 1

    return textlist, langlist

def nonen_clean_text_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    word2ph_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        phones_list.append(phones)
        if lang == "zh":
            word2ph_list.append(word2ph)
        norm_text_list.append(norm_text)
    print(word2ph_list)
    phones = sum(phones_list, [])
    word2ph = sum(word2ph_list, [])
    norm_text = ' '.join(norm_text_list)

    return phones, word2ph, norm_text

def nonen_get_bert_inf(text, language):
    if(language!="auto"):
        textlist, langlist = splite_en_inf(text, language)
    else:
        textlist=[]
        langlist=[]
        for tmp in LangSegment.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    bert_list = []
    for i in range(len(textlist)):
        text = textlist[i]
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(text, lang)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)

    return bert

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def get_cleaned_text_fianl(text,language):
    if language in {"en","all_zh","all_ja"}:
        phones, word2ph, norm_text = clean_text_inf(text, language)
    elif language in {"zh", "ja","auto"}:
        phones, word2ph, norm_text = nonen_clean_text_inf(text, language)
    return phones, word2ph, norm_text

def get_bert_final(phones, word2ph, norm_text, text_language, device, text):
    if text_language == "en":
        bert = get_bert_inf(phones, word2ph, norm_text, text_language)
    elif text_language in {"zh", "ja","auto"}:
        bert = nonen_get_bert_inf(text, text_language)
    elif text_language == "all_zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros((1024, len(phones))).to(device)
    return bert

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)

def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)

def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])

def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])

# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：]'
    items = re.split(f'({punds})', inp)
    items = ["".join(group) for group in zip(items[::2], items[1::2])]
    opt = "\n".join(items)
    return opt

class GPT_SoVITS:
    def __init__(self):
        self.model = None
        # is_half = True
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, gpt_path, sovits_path):
        self.hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if is_half == True:
            t2s_model = t2s_model.half()
        self.t2s_model = t2s_model.to(device)
        self.t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        if ("pretrained" not in sovits_path):
            del vq_model.enc_q
        if is_half == True:
            self.vq_model = vq_model.half().to(device)
        else:
            self.vq_model = vq_model.to(device)
        self.vq_model.eval()
        print(self.vq_model.load_state_dict(dict_s2["weight"], strict=False))
    
    def predict(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="不切", save_path = 'vits_res.wav'):
        print(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut)
        return self.get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, save_path)

    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="不切", save_path = 'vits_res.wav'):
        t0 = ttime()
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        text = text.strip("\n")
        if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
        print("实际输入的参考文本:", prompt_text)
        print("实际输入的目标文本:", text)
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if is_half == True else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = self.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
        t1 = ttime()
        
        dict_language = {
            "中文": "all_zh",#全部按中文识别
            "英文": "en",#全部按英文识别#######不变
            "日文": "all_ja",#全部按日文识别
            "中英混合": "zh",#按中英混合识别####不变
            "日英混合": "ja",#按日英混合识别####不变
            "多语种混合": "auto",#多语种启动切分识别语种
        }
        prompt_language = dict_language[prompt_language]
        text_language = dict_language[text_language]

        phones1, word2ph1, norm_text1=get_cleaned_text_fianl(prompt_text, prompt_language)

        if (how_to_cut == "凑四句一切"):
            text = cut1(text)
        elif (how_to_cut == "凑50字一切"):
            text = cut2(text)
        elif (how_to_cut == "按中文句号。切"):
            text = cut3(text)
        elif (how_to_cut == "按英文句号.切"):
            text = cut4(text)
        elif (how_to_cut == "按标点符号切"):
            text = cut5(text)
        text = text.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
        print("实际输入的目标文本(切句后):", text)
        texts = text.split("\n")
        audio_opt = []
        bert1=get_bert_final(phones1, word2ph1, norm_text1, prompt_language, device, text).to(dtype)

        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in splits): text += "。" if text_language != "en" else "."
            print("实际输入的目标文本(每句):", text)
            phones2, word2ph2, norm_text2 = get_cleaned_text_fianl(text, text_language)
            bert2 = get_bert_final(phones2, word2ph2, norm_text2, text_language, device, text).to(dtype)

            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
            prompt = prompt_semantic.unsqueeze(0).to(device)
            t2 = ttime()
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=self.config["inference"]["top_k"],
                    early_stop_num=self.hz * self.max_sec,
                )
            t3 = ttime()
            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(
                0
            )  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = get_spepc(self.hps, ref_wav_path)  # .to(device)
            if is_half == True:
                refer = refer.half().to(device)
            else:
                refer = refer.to(device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = (
                self.vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
                )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            max_audio=np.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio/=max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        # yield self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        #     np.int16
        # )
        write(save_path, self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16))
        return save_path
if __name__ == "__main__":
    GPT_SoVITS_inference = GPT_SoVITS()
    gpt_path = "GPT_SoVITS/pretrained_models/Gnews-e15.ckpt"
    sovits_path = "GPT_SoVITS/pretrained_models/Gnews_e8_s96.pth"
    GPT_SoVITS_inference.load_model(gpt_path, sovits_path)
    ref_wav_path = "GPT_SoVITS/reference_wav/Gnews/Gnews.mp3_0000270720_0000424960.wav"
    # 参考音频的文本
    from ASR import WhisperASR, FunASR
    asr = FunASR()
    prompt_text = ""
    prompt_text = asr.transcribe(ref_wav_path)
    prompt_language = "中文"
    text = "大家好，这是我语音克隆的声音，本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE."
    text_language = "中英混合" 
    how_to_cut = "不切" # ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"]
    print("参考音频文本：", prompt_text)
    print("目标文本：", text)
    save_audio_file = "./result.wav"
    GPT_SoVITS_inference.predict(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, save_audio_file)