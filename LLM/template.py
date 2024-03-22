from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMTemplate:
    def __init__(self, model_name_or_path, mode='offline'):
        self.mode = mode
        self.model, self.tokenizer = self.init_model(model_name_or_path)
        self.history = None
    
    def init_model(self, model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                     device_map="auto", 
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return model, tokenizer   
    
    def generate(self, prompt, system_prompt=""):
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
