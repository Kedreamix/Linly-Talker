'''
pip install -U g4f[all]
https://github.com/xtekky/gpt4free
'''
from g4f.client import Client

class GPT4FREE:
    def __init__(self, prefix_prompt = '''请用少于25个字回答以下问题\n\n'''):
        self.client = Client()
        self.prefix_prompt = prefix_prompt
        self.history = []
        '''
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            ...
        )
        print(response.choices[0].message.content)
        '''
    def generate(self, question, system_prompt="You are a helpful assistant."):
        self.history += [{
                "role": "user", 
                "content": self.prefix_prompt + question
            }]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages= [{"role": "system", 
                        "content": system_prompt}] 
                        if system_prompt else [] + self.history,
        )
        
        answer = response.choices[0].message.content
        if 'sorry' in answer.lower():
            return '对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n'
        
        return response.choices[0].message.content
    def chat(self, system_prompt = "You are a helpful assistant.", message = "", history=[]):
        response = self.generate(message, system_prompt)
        self.history += [{
            "role": "assistants",
            "content": response
        }]
        history.append((message, response))
        return response, history
    def clear_history(self):
        # 清空历史记录
        self.history = []


def test():
    llm = GPT4FREE()
    answer, history = llm.chat("", "如何应对压力？")
    print(answer, history)
    from time import sleep
    sleep(5)
    answer, history = llm.chat("", "能不能更加详细一点呢")
    print(answer, history)


if __name__ == '__main__':
    test()