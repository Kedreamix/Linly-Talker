from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
import shutil
import os
from loguru import logger
import sys
import gc
import torch
from typing import Optional
from fastapi.responses import FileResponse

sys.path.append("./")
app = FastAPI()

# Global variable to store the currently loaded Talker model
talker = None

USE_REF_VIDEO = False
REF_VIDEO = None
REF_INFO = 'pose'
USE_IDLE_MODE = False
AUDIO_LENGTH = 5

class TalkerRequest(BaseModel):
    preprocess_type: str = 'crop'
    is_still_mode: bool = False
    enhancer: bool = False
    batch_size: int = 4
    size_of_image: int = 256
    pose_style: int = 0
    facerender: str = 'facevid2vid'
    exp_weight: float = 1.0
    blink_every: bool = True
    talker_method: str = 'SadTalker'
    fps: int = 30

async def clear_memory():
    """Asynchronous function to clear GPU memory."""
    logger.info("Clearing GPU memory resources")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info(f"GPU memory usage after clearing: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

@app.post("/talker_change_model/")
async def change_model(model_name: str = Query(..., description="Name of the Talker model to load")):
    """Change digital human conversation model and load corresponding resources."""
    global talker

    # Clear memory to free up unnecessary resources before loading a new model
    await clear_memory()

    if model_name not in ['SadTalker', 'Wav2Lip', 'Wav2Lipv2', 'NeRFTalk']:
        raise HTTPException(status_code=400, detail="Other models are not integrated yet. Please wait for updates.")

    try:
        if model_name == 'SadTalker':
            from TFG import SadTalker
            talker = SadTalker(lazy_load=True)
            logger.info("SadTalker model loaded successfully")
        elif model_name == 'Wav2Lip':
            from TFG import Wav2Lip
            talker = Wav2Lip("checkpoints/wav2lip_gan.pth")
            logger.info("Wav2Lip model loaded successfully")
        elif model_name == 'Wav2Lipv2':
            from TFG import Wav2Lipv2
            talker = Wav2Lipv2('checkpoints/wav2lipv2.pth')
            logger.info("Wav2Lipv2 model loaded successfully, capable of generating higher quality results")
        elif model_name == 'NeRFTalk':
            from TFG import NeRFTalk
            talker = NeRFTalk()
            talker.init_model('checkpoints/Obama_ave.pth', 'checkpoints/Obama.json')
            logger.info("NeRFTalk model loaded successfully")
            logger.warning("NeRFTalk model is trained only for a single person, built-in with the Obama model, uploading other images is ineffective.")
    except Exception as e:
        logger.error(f"Failed to load {model_name} model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load {model_name} model: {e}")

    return {"message": f"{model_name} model loaded successfully"}

@app.post("/talker_response/")
async def talker_response(
    preprocess_type: str = Form('crop'),
    is_still_mode: bool = Form(False),
    enhancer: bool = Form(False),
    batch_size: int = Form(4),
    size_of_image: int = Form(256),
    pose_style: int = Form(0),
    facerender: str = Form('facevid2vid'),
    exp_weight: float = Form(1.0),
    blink_every: bool = Form(True),
    talker_method: str = Form('SadTalker'),
    fps: int = Form(30),
    source_image: UploadFile = File(..., description="The source image file"),
    driven_audio: UploadFile = File(..., description="The audio file that will drive the talking head"),
):
    """Handle digital human conversation requests and generate video."""
    global talker

    if talker is None:
        raise HTTPException(status_code=400, detail="Talker model not loaded. Please load a model first.")

    # Assemble the request data into the TalkerRequest model
    request = TalkerRequest(
        preprocess_type=preprocess_type,
        is_still_mode=is_still_mode,
        enhancer=enhancer,
        batch_size=batch_size,
        size_of_image=size_of_image,
        pose_style=pose_style,
        facerender=facerender,
        exp_weight=exp_weight,
        blink_every=blink_every,
        talker_method=talker_method,
        fps=fps,
    )
    # print(request)

    # Temporary file paths
    temp_image_path = "temp_source_image.jpg"
    temp_audio_path = "temp_driven_audio.wav"

    try:
        # Save uploaded files temporarily
        with open(temp_image_path, "wb") as image_file:
            shutil.copyfileobj(source_image.file, image_file)
        with open(temp_audio_path, "wb") as audio_file:
            shutil.copyfileobj(driven_audio.file, audio_file)

        # Video generation
        if request.talker_method == 'SadTalker':
            video_path = talker.test2(
                temp_image_path, 
                temp_audio_path,
                request.preprocess_type, 
                request.is_still_mode, 
                request.enhancer, 
                request.batch_size, 
                request.size_of_image, 
                request.pose_style, 
                request.facerender, 
                request.exp_weight, 
                REF_VIDEO, REF_INFO, USE_IDLE_MODE, AUDIO_LENGTH,
                request.blink_every, 
                request.fps,
            )
        elif request.talker_method == 'Wav2Lip':
            video_path = talker.predict(temp_image_path, temp_audio_path, request.batch_size)
        elif request.talker_method == 'Wav2Lipv2':
            video_path = talker.run(temp_image_path, temp_audio_path, request.batch_size)
        elif request.talker_method == 'NeRFTalk':
            video_path = talker.predict(temp_audio_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported method")

        # Ensure the video file exists and return it
        if os.path.exists(video_path):
            return FileResponse(video_path, media_type='video/mp4', filename=os.path.basename(video_path))
        else:
            raise HTTPException(status_code=404, detail="Video file not found")
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
