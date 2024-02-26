'''
pip install openai
'''
import os
import openai

class ChatGPT():
    def __init__(self, model_path = 'gpt-3.5-turbo', api_key = None, proxy_url = None):
        if proxy_url:
            os.environ['https_proxy'] = proxy_url if proxy_url else None
            os.environ['http_proxy'] = proxy_url if proxy_url else None
        openai.api_key = api_key
        self.model_path = model_path

    def generate(self, message):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_path,
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"