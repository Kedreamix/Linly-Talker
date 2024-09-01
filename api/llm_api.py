from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from loguru import logger
import gc, torch
import sys
sys.path.append('./')
app = FastAPI()

# 全局变量用于存储当前加载的LLM模型
from LLM import LLM
llm_class = LLM(mode='offline')

# 默认不使用LLM模型，直接回复问题，同时减少显存占用！
llm = llm_class.init_model('直接回复 Direct Reply')

# 默认系统提示语
PREFIX_PROMPT = '请用少于25个字回答以下问题\n\n'
PREFIX_PROMPT = ''

DEFAULT_SYSTEM = '你是一个很有帮助的助手'

class LLMRequest(BaseModel):
    question: str = '请问什么是FastAPI？'
    model_name: str = 'Linly'
    gemini_apikey: str = ''  # Gemini模型的API密钥
    openai_apikey: str = ''  # OpenAI的API密钥
    proxy_url: str = None  # 代理URL

@app.post("/llm_change_model/")
async def change_model(
    model_name: str = Query(..., description="要加载的LLM模型名称"),
    gemini_apikey: str = Query('', description="Gemini API 密钥"),
    openai_apikey: str = Query('', description="OpenAI API 密钥"),
    proxy_url: str = Query(None, description="代理 URL")
):
    """更换LLM模型并加载相应资源。"""
    global llm

    # 清理显存（具体实现依赖于模型库）
    await clear_memory()

    try:
        if model_name == 'Linly':
            llm = llm_class.init_model('Linly', 'Linly-AI/Chinese-LLaMA-2-7B-hf', prefix_prompt=PREFIX_PROMPT)
            logger.info("Linly模型导入成功")
        elif model_name in ['Qwen', 'Qwen2']:
            model_path = 'Qwen/Qwen-1_8B-Chat' if model_name == 'Qwen' else 'Qwen/Qwen1.5-0.5B-Chat'
            llm = llm_class.init_model(model_name, model_path, prefix_prompt=PREFIX_PROMPT)
            logger.info(f"{model_name} 模型导入成功")
        elif model_name == 'Gemini':
            if gemini_apikey:
                llm = llm_class.init_model('Gemini', 'gemini-pro', gemini_apikey, proxy_url)
                logger.info("Gemini模型导入成功")
            else:
                raise HTTPException(status_code=400, detail="请填写Gemini的API密钥")
        elif model_name == 'ChatGLM':
            llm = llm_class.init_model('ChatGLM', 'THUDM/chatglm3-6b', prefix_prompt=PREFIX_PROMPT)
            logger.info("ChatGLM模型导入成功")
        elif model_name == 'ChatGPT':
            if openai_apikey:
                llm = llm_class.init_model('ChatGPT', api_key=openai_apikey, proxy_url=proxy_url, prefix_prompt=PREFIX_PROMPT)
                logger.info("ChatGPT模型导入成功")
            else:
                raise HTTPException(status_code=400, detail="请填写OpenAI的API密钥")
        elif model_name == 'GPT4Free':
            llm = llm_class.init_model('GPT4Free', prefix_prompt=PREFIX_PROMPT)
            logger.info("GPT4Free模型导入成功，请注意该模型可能不稳定")
        elif model_name == '直接回复 Direct Reply':
            llm = llm_class.init_model(model_name)
            logger.info("直接回复模式激活，不使用LLM模型")
        else:
            raise HTTPException(status_code=400, detail=f"未知的LLM模型: {model_name}")
    except Exception as e:
        logger.error(f"{model_name}模型加载失败: {e}")
        raise HTTPException(status_code=500, detail=f"{model_name}模型加载失败: {e}")

    return {"message": f"{model_name} 模型加载成功"}

@app.post("/llm_response/")
async def llm_response(request: LLMRequest):
    """处理LLM模型的问答请求。"""
    global llm

    if not request.question:
        raise HTTPException(status_code=400, detail="问题内容不能为空")

    if llm is None:
        raise HTTPException(status_code=400, detail="LLM模型未加载，请先加载模型")

    try:
        answer = llm.generate(request.question, DEFAULT_SYSTEM)
        logger.info(f"LLM 回复：{answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"处理LLM请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理LLM请求失败: {e}")

async def clear_memory():
    """清理显存的异步函数"""
    logger.info("清理显存资源")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info(f"显存使用情况: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)