from .Linly import Linly
from .Qwen import Qwen
from .Qwen2 import Qwen2
try:
    from .Gemini import Gemini
except Exception as e:
    print("Gemini模型加载失败，可能是因为没有google-generativeai库，但是Gemini模型不是必顂的，可以忽略")
from .ChatGPT import ChatGPT
from .ChatGLM import ChatGLM
from .Llama2Chinese import Llama2Chinese
from .GPT4Free import GPT4FREE
from .QAnything import QAnything

def test_Linly(question = "如何应对压力？", mode='offline', model_path="Linly-AI/Chinese-LLaMA-2-7B-hf"):
    llm = Linly(mode, model_path)
    answer = llm.generate(question)
    print(answer)
    
def test_Qwen(question = "如何应对压力？", mode='offline', model_path="Qwen/Qwen-1_8B-Chat"):
    llm = Qwen(mode, model_path)
    answer = llm.generate(question)
    print(answer)
    
def test_Gemini(question = "如何应对压力？", model_path='gemini-pro', api_key=None, proxy_url=None):
    llm = Gemini(model_path, api_key, proxy_url)
    answer = llm.generate(question)
    print(answer)
    
def test_ChatGPT(question = "如何应对压力？", model_path = 'gpt-3.5-turbo', api_key = None, proxy_url = None):
    llm = ChatGPT(model_path, api_key, proxy_url)
    answer = llm.generate(question)
    print(answer)
    
class LLM:
    def __init__(self, mode='offline'):
        self.mode = mode
        
    def init_model(self, model_name, model_path='', api_key=None, proxy_url=None, prefix_prompt='''请用少于25个字回答以下问题\n\n'''):
        if model_name not in ['Linly', 'Qwen', 'Qwen2', 'Gemini', 'ChatGLM', 'ChatGPT', 'Llama2Chinese', 'GPT4Free', 'QAnything', '直接回复 Direct Reply']:
            raise ValueError("model_name must be one of ['Linly', 'Qwen', 'Qwen2', 'Gemini', 'ChatGLM', 'ChatGPT', 'Llama2Chinese', 'GPT4Free', 'QAnything', '直接回复 Direct Reply']")
        if model_name == 'Linly':
            llm = Linly(self.mode, model_path)
        elif model_name == 'Qwen':
            llm = Qwen(self.mode, model_path)
        elif model_name == 'Qwen2':
            llm = Qwen2(self.mode, model_path)
        elif model_name == 'Gemini':
            llm = Gemini(model_path, api_key, proxy_url)
        elif model_name == 'ChatGLM':
            llm = ChatGLM(self.mode, model_path)
        elif model_name == 'ChatGPT':
            llm = ChatGPT(model_path, api_key, proxy_url)
        elif model_name == 'Llama2Chinese':
            llm = Llama2Chinese(model_path, self.mode)
        elif model_name == 'GPT4Free':
            llm = GPT4FREE()
        elif model_name == 'QAnything':
            llm = QAnything()
        elif model_name == '直接回复 Direct Reply':
            llm = self
        llm.prefix_prompt = prefix_prompt
        return llm
    
    def chat(self, system_prompt, message, history):
        response = self.generate(message, system_prompt)
        history.append((message, response))
        return response, history

    def generate(self, question, system_prompt = 'system无效'):
        return question
    
    def test_Linly(self, question="如何应对压力？", model_path="Linly-AI/Chinese-LLaMA-2-7B-hf"):
        llm = Linly(self.mode, model_path)
        answer = llm.generate(question)
        print(answer)

    def test_Qwen(self, question="如何应对压力？", model_path="Qwen/Qwen-1_8B-Chat"):
        llm = Qwen(self.mode, model_path)
        answer = llm.generate(question)
        print(answer)

    def test_Gemini(self, question="如何应对压力？", model_path='gemini-pro', api_key=None, proxy_url=None):
        llm = Gemini(model_path, api_key, proxy_url)
        answer = llm.generate(question)
        print(answer)
    
    def test_ChatGPT(self, question="如何应对压力？", model_path = 'gpt-3.5-turbo', api_key = None, proxy_url = None):
        llm = ChatGPT(model_path, api_key, proxy_url)
        answer = llm.generate(question)
        print(answer)
        
    def test_ChatGLM(self, question="如何应对压力？", model_path="THUDM/chatglm-6b"):
        llm = ChatGLM(mode=self.mode, model_name_or_path=model_path)
        answer = llm.generate(question)
        print(answer)

if __name__ == '__main__':
    llm_class = LLM(mode='offline')
    llm_class.init_model('直接回复 Direct Reply')
    question = '如何应对压力？'
    answer = llm_class.generate(question)
    # llm.test_Qwen()
    # llm.test_Linly()
    # llm.test_Gemini()
    # llm.test_ChatGLM()