# TTS 赋予数字人真实的语音交互能力

Edge-TTS是一个Python库，它使用微软的Azure Cognitive Services来实现文本到语音转换（TTS）。

该库提供了一个简单的API，可以将文本转换为语音，并且支持多种语言和声音。要使用Edge-TTS库，首先需要安装上Edge-TTS库，安装直接使用pip 进行安装即可。

```bash
pip install -U edge-tts
```

> 如果想更细究使用方式，可参考[https://github.com/rany2/edge-tts](https://github.com/rany2/edge-tts)



根据源代码，我编写了一个 `EdgeTTS` 的类，能够更好的使用，并且增加了保存字幕文件的功能，能增加体验感

```python
class EdgeTTS:
    def __init__(self, list_voices = False, proxy = None) -> None:
        voices = list_voices_fn(proxy=proxy)
        self.SUPPORTED_VOICE = [item['ShortName'] for item in voices]
        self.SUPPORTED_VOICE.sort(reverse=True)
        if list_voices:
            print(", ".join(self.SUPPORTED_VOICE))

    def preprocess(self, rate, volume, pitch):
        if rate >= 0:
            rate = f'+{rate}%'
        else:
            rate = f'{rate}%'
        if pitch >= 0:
            pitch = f'+{pitch}Hz'
        else:
            pitch = f'{pitch}Hz'
        volume = 100 - volume
        volume = f'-{volume}%'
        return rate, volume, pitch

    def predict(self,TEXT, VOICE, RATE, VOLUME, PITCH, OUTPUT_FILE='result.wav', OUTPUT_SUBS='result.vtt', words_in_cue = 8):
        async def amain() -> None:
            """Main function"""
            rate, volume, pitch = self.preprocess(rate = RATE, volume = VOLUME, pitch = PITCH)
            communicate = Communicate(TEXT, VOICE, rate = rate, volume = volume, pitch = pitch)
            subs: SubMaker = SubMaker()
            sub_file: Union[TextIOWrapper, TextIO] = (
                open(OUTPUT_SUBS, "w", encoding="utf-8")
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # audio_file.write(chunk["data"])
                    pass
                elif chunk["type"] == "WordBoundary":
                    # print((chunk["offset"], chunk["duration"]), chunk["text"])
                    subs.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])
            sub_file.write(subs.generate_subs(words_in_cue))
            await communicate.save(OUTPUT_FILE)
            
        
        # loop = asyncio.get_event_loop_policy().get_event_loop()
        # try:
        #     loop.run_until_complete(amain())
        # finally:
        #     loop.close()
        asyncio.run(amain())
        with open(OUTPUT_SUBS, 'r', encoding='utf-8') as file:
            vtt_lines = file.readlines()

        # 去掉每一行文字中的空格
        vtt_lines_without_spaces = [line.replace(" ", "") if "-->" not in line else line for line in vtt_lines]
        # print(vtt_lines_without_spaces)
        with open(OUTPUT_SUBS, 'w', encoding='utf-8') as output_file:
            output_file.writelines(vtt_lines_without_spaces)
        return OUTPUT_FILE, OUTPUT_SUBS
```



同时在`src`文件夹下，写了一个简易的`WebUI`

```bash
python app.py
```

![TTS](../docs/TTS.png)

