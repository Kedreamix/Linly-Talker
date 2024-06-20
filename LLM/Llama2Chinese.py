import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Llama2Chinese:
    def __init__(self, model_path, mode='offline'):
        """
        初始化LLM模板

        Args:
            model_name_or_path (str): 模型名称或路径
            mode (str, optional): 模式，'offline'表示离线模式，'api'表示使用API模式。默认为'offline'。
        """
        self.mode = mode
        self.load_in_8bit = True
        self.prefix_prompt = '''请用少于25个字回答以下问题 '''
        self.history = []
        self.model, self.tokenizer = self.init_model(model_path)
        self.model.eval()    
    
    def init_model(self, model_path):
        """
        初始化语言模型

        Args:
            model_name_or_path (str): 模型名称或路径

        Returns:
            model: 加载的语言模型
            tokenizer: 加载的tokenizer
        """
        tokenizer = LlamaTokenizer.from_pretrained(model_path)

        base_model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='cuda:0',
        )
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)
        return base_model, tokenizer
    
    def generate(self, prompt, system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request."):
        """
        生成对话响应

        Args:
            prompt (str): 对话的提示
            system_prompt (str, optional): 系统提示。默认为""。

        Returns:
            str: 对话响应
        """
        device = torch.device(0)
        # TODO: 模型预测
        # 这一块需要尤其注意，这里的模板是借鉴了HuggingFace上的一些推理模板，需要根据自己的模型进行调整
        # 这里的模板主要是为了方便调试，因为模型预测的时候，会有很多不同的输入，所以可以根据自己的模型进行调整
        if self.mode != 'api':
            try:
                # max_memory = 1024
                question = self.message_to_prompt(prompt, system_prompt)
                # print(question)
                # if len(question) > max_memory:
                #     question = question[-max_memory:]
                inputs = self.tokenizer(question, return_tensors="pt")
                # input_ids = inputs["input_ids"].to(device)
                generation_config = dict(
                    temperature=0.5,
                    top_k=40,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,
                    repetition_penalty=1.1,
                    max_new_tokens=512
                    )
                generate_ids = self.model.generate(
                    input_ids = inputs["input_ids"].to(device),
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generation_config
                )
                response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                response = response.split("### Response:")[-1].strip()

                # response, self.history = self.model.chat(self.tokenizer, prompt, history=self.history, system = system_prompt)
                return response
            except Exception as e:
                print(e)
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        else:
            return self.predict_api(prompt)
    
    def message_to_prompt(self, message, system_prompt=""):
        system_prompt = self.prefix_prompt + system_prompt
        for interaction in self.history:
            user_prompt, bot_prompt = str(interaction[0]).strip(' '), str(interaction[1]).strip(' ')
            system_prompt = f"{system_prompt} ### Instruction:\n{user_prompt}\n\n### Response: {bot_prompt}\n\n"
        prompt = f"{system_prompt} ### Instruction:\n{message.strip()}\n\n### Response: "
        return prompt
    
    def predict_api(self, prompt):
        """
        使用API预测对话响应

        Args:
            prompt (str): 对话的提示

        Returns:
            str: 对话响应
        """
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        pass 
    
    def chat(self, system_prompt, message):
        response = self.generate(message, system_prompt)
        self.history.append((message, response))
        return response, self.history
    
    def clear_history(self):
        self.history = []

def test():
    llm = Llama2Chinese("./Llama2-chat-13B-Chinese-50W")
    answer = llm.generate("如何应对压力")
    print(answer)
    
if __name__ == '__main__':
    test()
