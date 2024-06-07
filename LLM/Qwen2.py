import os
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Qwen2:
    def __init__(self, mode='offline', model_path="Qwen/Qwen1.5-0.5B-Chat", prefix_prompt = '''请用少于25个字回答以下问题\n\n'''):
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
    
    def init_model(self, path = "Qwen/Qwen2-0.5B"):
        model = AutoModelForCausalLM.from_pretrained(path, 
                                                     device_map="auto", 
                                                     torch_dtype="auto",
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        return model, tokenizer   
    
    def generate(self, question, system_prompt="You are a helpful assistant."):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.mode != 'api':
            try:
                # response, self.history = self.model.chat(self.tokenizer, self.data["question"], history=self.history, system = system_prompt)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.prefix_prompt + question}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
    llm = Qwen2(mode='offline', model_path="Qwen/Qwen1.5-0.5B-Chat")
    # llm = Qwen2(mode='offline', model_path="Qwen/Qwen2-0.5B")
    answer = llm.generate("如何应对压力？")
    print(answer)

if __name__ == '__main__':
    test()
