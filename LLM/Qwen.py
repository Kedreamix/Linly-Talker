import os
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Qwen:
    def __init__(self, mode='offline', model_path="Qwen/Qwen-1_8B-Chat", prefix_prompt = '''请用少于25个字回答以下问题\n\n'''):
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        self.url = "http://ip:port" # local server: http://ip:port
        self.headers = {
            "Content-Type": "application/json"
        }
        self.data = {
            "question": "北京有什么好玩的地方？"
        }
        self.prefix_prompt = prefix_prompt
        self.mode = mode
        self.model, self.tokenizer = self.init_model(model_path)
        self.history = None
    
    def init_model(self, path = "Qwen/Qwen-1_8B-Chat"):
        model = AutoModelForCausalLM.from_pretrained(path, 
                                                     device_map="auto", 
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        return model, tokenizer   
    
    def generate(self, question, system_prompt=""):
        if self.mode != 'api':
            self.data["question"] = self.prefix_prompt + question
            try:
                response, self.history = self.model.chat(self.tokenizer, self.data["question"], history=self.history, system = system_prompt)
                # print(self.history)
                return response
            except Exception as e:
                print(e)
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        else:
            return self.predict_api(question)
    def predict_api(self, question):
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        pass 
    
    def chat(self, system_prompt, message, history):
        response = self.generate(message, system_prompt)
        history.append((message, response))
        return response, history
    
    def clear_history(self):
        # 清空历史记录
        self.history = []
    
def test():
    llm = Qwen(mode='offline', model_path="Qwen/Qwen-1_8B-Chat")
    answer = llm.generate("如何应对压力？")
    print(answer)

if __name__ == '__main__':
    test()
