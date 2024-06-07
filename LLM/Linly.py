import os
import torch
import requests
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import ip, api_port, model_path
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Linly:
    def __init__(self, mode='api', model_path="Linly-AI/Chinese-LLaMA-2-7B-hf", prefix_prompt = '''请用少于25个字回答以下问题\n\n'''):
        # mode = api need
        # 定义设置的api的服务器,首先记得运行Linly-api-fast.py 填入ip地址和端口号
        self.url = f"http://{ip}:{api_port}" # local server: http://ip:port
        self.headers = {
            "Content-Type": "application/json"
        }
        self.data = {
            "question": "北京有什么好玩的地方？"
        }
        # 全局设定的prompt
        self.prefix_prompt = prefix_prompt
        self.mode = mode
        if mode != 'api':
            self.model, self.tokenizer = self.init_model(model_path)
        self.history = []
    
    def init_model(self, path = "Linly-AI/Chinese-LLaMA-2-7B-hf"):
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda:0",
                                                    torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
        return model, tokenizer   
    
    def generate(self, question, system_prompt=""):
        if self.mode != 'api':
            self.data["question"] = self.message_to_prompt(question, system_prompt)
            inputs = self.tokenizer(self.data["question"], return_tensors="pt").to("cuda:0")
            try:
                generate_ids = self.model.generate(inputs.input_ids, 
                                                   max_new_tokens=2048, 
                                                   do_sample=True, 
                                                   top_k=20, 
                                                   top_p=0.84,
                                                   temperature=1, 
                                                   repetition_penalty=1.15, 
                                                   eos_token_id=2, 
                                                   bos_token_id=1,
                                                   pad_token_id=0)
                response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                response = response.split("### Response:")[-1]
                return response
            except:
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        elif self.mode == 'api':
            return self.predict_api(question)
    
    def message_to_prompt(self, message, system_prompt=""):
        system_prompt = self.prefix_prompt + system_prompt
        for interaction in self.history:
            user_prompt, bot_prompt = str(interaction[0]).strip(' '), str(interaction[1]).strip(' ')
            system_prompt = f"{system_prompt} User: {user_prompt} Bot: {bot_prompt}"
        prompt = f"{system_prompt} ### Instruction:{message.strip()}  ### Response:"
        return prompt
        
    def predict_api(self, question):
        # FastAPI Predict 调用API来进行预测
        self.data["question"] = question
        headers = {'Content-Type': 'application/json'}
        data = {"prompt": question}
        response = requests.post(url=self.url, headers=headers, data=json.dumps(data))
        return response.json()['response']
            
    def chat(self, system_prompt, message, history):
        self.history = history
        prompt = self.message_to_prompt(message, system_prompt)
        response = self.generate(prompt)
        self.history.append([message, response])
        return response, self.history
    
    def clear_history(self):
        # 清空历史记录
        self.history = []

def test():
    llm = Linly(mode='offline',model_path='../Linly-AI/Chinese-LLaMA-2-7B-hf')
    answer = llm.generate("如何应对压力？")
    print(answer)

if __name__ == '__main__':
    test()
