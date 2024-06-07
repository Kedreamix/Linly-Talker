'''
根据现有HuggingFace的LLM的调用方式写一个模板
注意一下是否调用方式类似，如果不类似，需要修改里面的推理代码
支持：Qwen，ChatLLM等
'''
from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatGLM:
    def __init__(self, mode='offline', model_path = 'THUDM/chatglm3-6b', prefix_prompt = '''请用少于25个字回答以下问题\n\n'''):
        self.mode = mode
        self.model, self.tokenizer = self.init_model(model_path)
        self.history = None
        self.prefix_prompt = prefix_prompt
        assert self.mode == 'offline', "ChatGLM只支持离线模式"
    
    def init_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     device_map="auto", 
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer   
    
    def generate(self, prompt, system_prompt=""):
        if self.mode != 'api':
            try:
                # 注意这里的history是个list，每次调用都会把prompt和response都放进去
                # 如果使用，查看对应的方法是否类似，这里是问答的重要部份，这里正确，基本就正确了
                response, self.history = self.model.chat(self.tokenizer, self.prefix_prompt + prompt, history=self.history)
                return response
            except Exception as e:
                print(e)
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        else:
            return self.predict_api(prompt)
    
    def predict_api(self, prompt):
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        pass 
    
    def chat(self, system_prompt, message):
        response = self.generate(message, system_prompt)
        self.history.append((message, response))
        return response, self.history
    
    def clear_history(self):
        self.history = []

def test():
    llm = ChatGLM(mode='offline',model_path='THUDM/chatglm3-6b')
    answer = llm.generate("如何应对压力？")
    print(answer)

if __name__ == '__main__':
    test()
