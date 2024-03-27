from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMTemplate:
    def __init__(self, model_name_or_path, mode='offline'):
        self.mode = mode
        # 模型初始化
        self.model, self.tokenizer = self.init_model(model_name_or_path)
        self.history = None
    
    def init_model(self, model_name_or_path):
        # TODO: 模型加载
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                     device_map="auto", 
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return model, tokenizer   
    
    def generate(self, prompt, system_prompt=""):
        # TODO: 模型预测
        # 这一块需要尤其注意，这里的模板是借鉴了HuggingFace上的一些推理模板，需要根据自己的模型进行调整
        # 这里的模板主要是为了方便调试，因为模型预测的时候，会有很多不同的输入，所以可以根据自己的模型进行调整
        if self.mode != 'api':
            try:
                response, self.history = self.model.chat(self.tokenizer, prompt, history=self.history, system = system_prompt)
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
