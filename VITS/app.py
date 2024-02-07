from GPT_SoVITS import *
import gradio as gr

GPT_SoVITS_inference = GPT_SoVITS()
gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
GPT_SoVITS_inference.load_model(gpt_path, sovits_path)

with gr.Blocks(title = "GPT-SoVITS") as demo:
    with gr.Row():
        gr.HTML("<center><h1>GPT-SoVITS WebUI</h1></center>")
    gr.Markdown(value="*请上传并填写参考信息")
    with gr.Row():
        inp_ref = gr.Audio(label="请上传3~10秒内参考音频，超过会报错！", sources=["microphone", "upload"], type="filepath")
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
    gr.Markdown(value="文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。")
    with gr.Row():
        text_inp = gr.Textbox(label="需要合成的切分前文本", value="")
        button1 = gr.Button("凑四句一切", variant="primary")
        button2 = gr.Button("凑50字一切", variant="primary")
        button3 = gr.Button("按中文句号。切", variant="primary")
        button4 = gr.Button("按英文句号.切", variant="primary")
        button5 = gr.Button("按标点符号切", variant="primary")
        text_opt = gr.Textbox(label="切分后文本", value="")
        button1.click(cut1, [text_inp], [text_opt])
        button2.click(cut2, [text_inp], [text_opt])
        button3.click(cut3, [text_inp], [text_opt])
        button4.click(cut4, [text_inp], [text_opt])
        button5.click(cut5, [text_inp], [text_opt])
    gr.Markdown(value="后续将支持混合语种编码文本输入。")

demo.launch()   