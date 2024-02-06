
from VITS import GPT_SoVITS
import gradio as gr

GPT_SoVITS_inference = GPT_SoVITS()
gpt_path = "../GPT-SoVITS/GPT_weights/yansang-e15.ckpt"
sovits_path = "../GPT-SoVITS/SoVITS_weights/yansang_e16_s144.pth"
GPT_SoVITS_inference.load_model(gpt_path, sovits_path)

with gr.Blocks(title = "GPT-SoVITS") as demo:
    gr.Markdown(value="*请上传并填写参考信息")
    with gr.Row():
        inp_ref = gr.Audio(label="请上传3~10秒内参考音频，超过会报错！", type="filepath")
        prompt_text = gr.Textbox(label="参考音频的文本", value="")
        prompt_language = gr.Dropdown(
            label="参考音频的语种", choices=["中文", "英文", "日文"], value="中文"
        )
    gr.Markdown(value="*请填写需要合成的目标文本。中英混合选中文，日英混合选日文，中日混合暂不支持，非目标语言文本自动遗弃。")
    with gr.Row():
        text = gr.Textbox(label="需要合成的文本", value="")
        text_language = gr.Dropdown(
            label="需要合成的语种", choices=["中文", "英文", "日文", "中英混合", "日英混合", "多语种混合"], value="中文"
        )
        how_to_cut = gr.Radio(
            label="怎么切",
            choices=["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切" ],
            value="凑四句一切",
            interactive=True,
        )
        inference_button = gr.Button("合成语音", variant="primary")
        output = gr.Audio(label="输出的语音")

    inference_button.click(
        GPT_SoVITS_inference.predict,
        [inp_ref, prompt_text, prompt_language, text, text_language, how_to_cut],
        [output],
    )

demo.launch()    

# if __name__ == "__main__":
    # ref_wav_path = "GPT-SoVITS/output/slicer_opt/vocal_output.wav_10.wav_0000846400_0000957760.wav"
    # prompt_text = "你为什么要一次一次的伤我的心啊？"
    # prompt_language = "中文"
    # text = "大家好，这是我语音克隆的声音，本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE."
    # text_language = "中英混合" 
    # how_to_cut = "不切" # ["不切", "凑四句一切", "凑50字一切", "按中文句号。切", "按英文句号.切", "按标点符号切"]
    # GPT_SoVITS_inference.predict(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut)