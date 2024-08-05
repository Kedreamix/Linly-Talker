import time
import requests
import json

def _extract_plain_response(data_string):
    '''
    从QAnything API返回的json数据中提取answer字段的值
    '''
    try:
        # 去掉字符串前后的双引号
        data_string = data_string.strip('"')
        
        # 将字符串转换为字典
        data_dict = json.loads(data_string.split(": ", 1)[1])
        
        # 提取并返回
        return data_dict.get("answer", "")
    except (json.JSONDecodeError, IndexError):
        return "Invalid format"

def _extract_dicts_from_data(json_string):
    '''
    从QAnything API返回的知识库信息(json数据)中提取data字段的多个字典(代表存在的各个知识库)
    '''
    try:
        # 将字符串转换为字典
        data_dict = json.loads(json_string)
        
        # 检查code是否为200
        if data_dict.get("code") == 200:
            # 提取并返回data中的多个字典
            return data_dict.get("data", [])
        else:
            return "Invalid code"
    except json.JSONDecodeError:
        return "Invalid JSON format"


class QAnything:
    def __init__(self, model_path: str='qanything', 
                 url_root: str='http://localhost:8777/api/local_doc_qa/',
                 url_chat_suffix: str='local_doc_chat',
                 url_kbs_suffix: str='list_knowledge_base',
                 default_kb_ids: list=["example_knowledge_base_id"]):  # NOTE: maybe expose to frontend
        """
        使用QAnything服务API基于知识库进行LLM对话, 需要至少一个知识库. 

        Args:
            model_path (str): 模型名称
            mode (str, optional): 模式，'offline'表示离线模式，'api'表示使用API模式。默认为'offline
            url_root (str, optional): 服务器的根URL。默认为'http://localhost:8777/api/local_doc_qa/'
            url_chat_suffix (str, optional): 服务器的对话接口后缀。默认为'local_doc_chat'
            url_kbs_suffix (str, optional): 服务器的知识库接口后缀。默认为'list_knowledge_base'
        """
        self.model_path = model_path
        self.url_root= url_root
        self.url_chat_suffix = url_chat_suffix
        self.url_kbs_suffix = url_kbs_suffix
        self.default_kb_ids = default_kb_ids
        kbs = self.get_kbs()
        all_kb_ids = [kb['kb_id'] for kb in kbs]
        kb_exist = all(kb_id in all_kb_ids for kb_id in self.default_kb_ids)
        if not kb_exist:
            print("默认的知识库ID不存在或连接失败，请检查")
            print(f"默认的知识库ID: {self.default_kb_ids}")
            print(f"所有的知识库ID: {all_kb_ids}")

    def get_kbs(self, user_id='zzp'):
        full_url = self.url_root + self.url_kbs_suffix
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "user_id": user_id
        }

        try:
            response = requests.post(full_url, headers=headers, data=json.dumps(data))
            return _extract_dicts_from_data(response.text)
        except Exception as e:
            print(f"QAnything list knowledge base: 请求发送失败: {e}")
            return []
    
    def send_request(self, prompt, user_id='zzp', kd_ids=None):
        full_url = self.url_root + self.url_chat_suffix
        headers = {
            'content-type': 'application/json'
        }
        kd_ids = kd_ids if kd_ids else self.default_kb_ids
        data = {
            "user_id": user_id,
            "kb_ids": kd_ids,
            "question": prompt,
        }
        try:
            start_time = time.time()
            response = requests.post(url=full_url, headers=headers, json=data, timeout=60)
            end_time = time.time()
            res = response.json()
            print(res['response'])
            print(f"响应状态码: {response.status_code}, 响应时间: {end_time - start_time}秒")
            return _extract_plain_response(res['response'])
        except Exception as e:
            print(f"QAnything chat: 请求发送失败: {e}")
    
    def generate(self, prompt):
        """
        生成对话响应

        Args:
            prompt (str): 对话的提示

        Returns:
            str: 对话响应
        """
        try:
            return self.send_request(prompt)
        except Exception as e:
            return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"


if __name__ == '__main__':
    llm = QAnything()
    answer = llm.generate("如何应对压力？")
    print(answer)
