from llama_cpp import Llama

class LlamacppChat:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # 使用GPU加速
            use_mlock=True,
            flash_attn=True,
            n_ctx=1024  # 设置上下文长度
        )

    def chat(self, messages):
        output = self.llm.create_chat_completion(
            messages=messages
        )
        return output["choices"][0]['message']['content']

# 使用示例
# if __name__ == "__main__":
#     model_path = 'Meta-Llama-3-8B-Instruct-Q4_K_M.gguf'  #https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
#     llama_chat = LlamacppChat(model_path)

#     messages = [
#         {
#             "role": "system",
#             "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary."
#         },
#         {
#             "role": "user",
#             "content": "介绍下你自己"
#         }
#     ]

#     response = llama_chat.chat(messages)
#     print(response)
