from src.Linly import Linly
from src.Qwen import Qwen
from src.Gemini import Gemini

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
    
class LLM:
    def __init__(self, mode='offline'):
        self.mode = mode
        
    def init_model(self, model_name, model_path, api_key=None, proxy_url=None):
        if model_name not in ['Linly', 'Qwen', 'Gemini']:
            raise ValueError("model_name must be 'Linly', 'Qwen', or 'Gemini'(其他模型还未集成)")
        if model_name == 'Linly':
            llm = Linly(self.mode, model_path)
        elif model_name == 'Qwen':
            llm = Qwen(self.mode, model_path)
        elif model_name == 'Gemini':
            llm = Gemini(model_path, api_key, proxy_url)
        return llm
    
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

if __name__ == '__main__':
    llm = LLM(mode='offline')
    llm.test_Qwen()
    # llm.test_Linly()
    # llm.test_Gemini()