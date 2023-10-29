import os
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import openbayes_serving as serv


class Predictor:
    def __init__(self):
        """
        负责加载相应的模型以及对元数据的初始化
        """
        self.model, self.tokenizer = self.init_model('./Chinese-LLaMA-2-7B-hf')
        self.prompt = '''请用少于25个字回答以下问题'''
        self.prompt = "" 
    def init_model(self, path = "Linly-AI/Chinese-LLaMA-2-7B-hf"):
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda:0",
                                                    torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
        return model, tokenizer

    def predict(self, json):
        """
        接受 HTTP 请求的内容，进行必要的处理后预测结果，最终将结果返回给调用方
        """
        question = json['question']
        if self.prompt:
            question = f"{self.prompt} ### Instruction:{question}  ### Response:"
        else:
            question = f"### Instruction:{question}  ### Response:"
        inputs = self.tokenizer(question, return_tensors="pt").to("cuda:0")
        try:
            generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=2048, do_sample=True, top_k=20, top_p=0.84,
                                        temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1,
                                        pad_token_id=0)
            response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print('log:', response)
            response = response.split("### Response:")[-1]
            return [response], "success"
        except:
            return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n", "fail"



if __name__ == '__main__':
    # /home/cvi_demo/anaconda3/envs/talk/lib/python3.8/site-packages/openbayes_serving/serv.py
    # nohup python Linly-api.py > api.out &
    serv.run(Predictor)