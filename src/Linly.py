import os
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Linly:
    def __init__(self, type='api', model_path="Linly-AI/Chinese-LLaMA-2-7B-hf") -> None:
        self.url = "https://P01son-52nfefhaaova.serv-c1.openbayes.net"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.data = {
            "question": "北京有什么好玩的地方？"
        }
        self.prompt = '''请用少于25个字回答以下问题'''
        self.type = type
        if type != 'api':
            self.model, self.tokenizer = self.init_model(model_path)
    
    def init_model(self, path = "Linly-AI/Chinese-LLaMA-2-7B-hf"):
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda:0",
                                                    torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
        return model, tokenizer   
    
    def generate(self, question):
        if self.type != 'api':
            self.data["question"] = f"{self.prompt} ### Instruction:{question}  ### Response:"
            inputs = self.tokenizer(self.data["question"], return_tensors="pt").to("cuda:0")
            try:
                generate_ids = self.model.generate(inputs.input_ids, max_new_tokens=2048, do_sample=True, top_k=20, top_p=0.84,
                                            temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1,
                                            pad_token_id=0)
                response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print('log:', response)
                response = response.split("### Response:")[-1]
                return response
            except:
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        else:
            return self.predict(question)
    
    def predict(self, question):

        self.data["question"] = f"{self.prompt} ### Instruction:{question}  ### Response:"
        response = requests.post(self.url, headers=self.headers, json=self.data)
        self.json = response.json()
        answer, tag = self.json
        if tag == 'success':
            return answer[0]
        else:
            return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"

def test():
    #llm = Linly(type='api')
    #answer = llm.predict("如何应对压力？")
    #print(answer)
    
    llm = Linly(type='offline',model_path="/home/cvi_demo/PythonProject/Talk/Chinese-LLaMA-2-7B-hf/")
    answer = llm.generate("如何应对压力？")
    print(answer)

# test()
