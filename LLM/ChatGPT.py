'''
pip install openai
'''
import os
import openai

class ChatGPT():
    def __init__(self, model_path = 'gpt-3.5-turbo', api_key = None, proxy_url = None, prefix_prompt = '''请用少于25个字回答以下问题\n\n'''):
        if proxy_url:
            os.environ['https_proxy'] = proxy_url if proxy_url else None
            os.environ['http_proxy'] = proxy_url if proxy_url else None
        openai.api_key = api_key
        self.model_path = model_path
        self.prefix_prompt = prefix_prompt

    def generate(self, message):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_path,
                messages=[
                    {"role": "user", "content": self.prefix_prompt + message}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        
if __name__ == '__main__':
    API_KEY = '******'
    # 若使用ChatGPT，要保证自己的APIKEY可用，并且服务器可访问OPENAI
    llm = ChatGPT(model_path='gpt-3.5-turbo', api_key=API_KEY, proxy_url=None)
    answer = llm.generate("如何应对压力？")
    print(answer)