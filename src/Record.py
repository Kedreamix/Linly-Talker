import pyaudio
import wave
import audioop

class Record:
    def __init__(self, output_file, silence_threshold=1000):
        self.output_file = output_file
        self.silence_threshold = silence_threshold

        # 设置录音参数
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        # 打开音频流
        stream = self.audio.open(format=self.format,
                                channels=self.channels,
                                rate=self.sample_rate,
                                input=True,
                                frames_per_buffer=self.chunk)

        print("开始录音...")

        frames = []
        silence_counter = 0

        # 录制音频
        while True:
            data = stream.read(self.chunk)
            frames.append(data)

            # 计算音频能量（RMS）
            rms = audioop.rms(data, 2)  # 2表示样本的宽度为2字节（16位）

            if rms < self.silence_threshold:
                silence_counter += 1
            else:
                silence_counter = 0

            # 如果连续一段时间都检测到静音，则认为录音结束
            if silence_counter >= 100:  # 假设每个chunk为10毫秒，连续1秒的静音
                break

        print("录音结束.")

        # 关闭流
        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        # 将录制的音频保存到WAV文件
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))


def test():
    # 创建Record对象并开始录音
    output_file = "recording.wav"  # 输出文件名
    silence_threshold = 500  # 静音阈值

    recorder = Record(output_file, silence_threshold)
    recorder.start_recording()

# test()