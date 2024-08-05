#!/bin/bash

# 提示用户选择下载方式
echo "Please select a model download method:"
echo "请选择模型下载方式："
echo "1. Download from ModelScope (No resuming capability)"
echo "1. 从 ModelScope 下载（有断点续传功能）推荐"
# 记得先下载modelscope, pip install modelscope
echo "记得先下载modelscope, pip install modelsope"
echo "2. Download from Huggingface (With resuming capability)"
echo "2. 从 Huggingface 下载（有断点续传功能）"
echo "3. Download from Huggingface mirror site (Possibly faster)"
echo "3. 从 Huggingface 镜像站点下载（镜像可能快一点）"
read -p "Enter 1, 2, or 3 to choose a download method: " download_option

# 下载模型
# Download the models
case $download_option in
  1)
    echo "Downloading models from ModelScope..."
    echo "正在从 ModelScope 下载模型..."
    python scripts/modelscope_download.py
    if [ $? -ne 0 ]; then
      echo "Failed to download models from ModelScope. Please check the scripts/modelscope_download.py script or your network connection."
      echo "从 ModelScope 下载模型失败，请检查脚本 scripts/modelscope_download.py 或网络连接。"
      exit 1
    fi
    ;;
  2)
    echo "Downloading models from Huggingface..."
    echo "正在从 Huggingface 下载模型..."
    python scripts/huggingface_download.py
    if [ $? -ne 0 ]; then
      echo "Failed to download models from Huggingface. Please check the scripts/huggingface_download.py script or your network connection."
      echo "从 Huggingface 下载模型失败，请检查脚本 scripts/huggingface_download.py 或网络连接。"
      exit 1
    fi
    ;;
  3)
    echo "Downloading models from Huggingface mirror site..."
    echo "正在从 Huggingface 镜像站点下载模型..."
    export HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download --resume-download --local-dir-use-symlinks False Kedreamix/Linly-Talker --local-dir Linly-Talker
    if [ $? -ne 0 ]; then
      echo "Failed to download models from Huggingface mirror site. Please check HF_ENDPOINT or your network connection."
      echo "从 Huggingface 镜像站点下载模型失败，请检查 HF_ENDPOINT 或网络连接。"
      exit 1
    fi
    ;;
  *)
    echo "Invalid selection. Please enter 1, 2, or 3."
    echo "无效选择，请输入 1, 2 或 3。"
    exit 1
    ;;
esac

echo "Model download completed."
echo "模型下载完成。"

# 检查并移动模型
# Check and move the models

# 2. 移动所有模型到当前目录
# Move all models to the current directory
if [ -d "Kedreamix/Linly-Talker/checkpoints" ]; then
  mv Kedreamix/Linly-Talker/checkpoints/* ./checkpoints
  if [ $? -ne 0 ]; then
    echo "Failed to move checkpoints."
    echo "移动 checkpoints 失败。"
    exit 1
  fi
else
  echo "Directory Kedreamix/Linly-Talker/checkpoints does not exist, cannot move checkpoints."
  echo "目录 Kedreamix/Linly-Talker/checkpoints 不存在，无法移动 checkpoints。"
  exit 1
fi

# 3. 若使用GFPGAN增强，安装对应的库
# If using GFPGAN enhancement, install the corresponding library
# pip install gfpgan

if [ -d "Kedreamix/Linly-Talker/gfpgan" ]; then
  mv Kedreamix/Linly-Talker/gfpgan ./
  if [ $? -ne 0 ]; then
    echo "Failed to move gfpgan directory."
    echo "移动 gfpgan 目录失败。"
    exit 1
  fi
else
  echo "Directory Kedreamix/Linly-Talker/gfpgan does not exist, cannot move gfpgan."
  echo "目录 Kedreamix/Linly-Talker/gfpgan 不存在，无法移动 gfpgan。"
  exit 1
fi

# 4. 语音克隆模型
# Voice cloning model
if [ -d "Kedreamix/Linly-Talker/GPT_SoVITS/pretrained_models" ]; then
  mv Kedreamix/Linly-Talker/GPT_SoVITS/pretrained_models/* ./GPT_SoVITS/pretrained_models/
  if [ $? -ne 0 ]; then
    echo "Failed to move GPT_SoVITS/pretrained_models."
    echo "移动 GPT_SoVITS/pretrained_models 失败。"
    exit 1
  fi
else
  echo "Directory Kedreamix/Linly-Talker/GPT_SoVITS/pretrained_models does not exist, cannot move pretrained models."
  echo "目录 Kedreamix/Linly-Talker/GPT_SoVITS/pretrained_models 不存在，无法移动预训练模型。"
  exit 1
fi

# 5. Qwen大模型
# Qwen model
if [ -d "Kedreamix/Linly-Talker/Qwen" ]; then
  mv Kedreamix/Linly-Talker/Qwen ./
  if [ $? -ne 0 ]; then
    echo "Failed to move Qwen directory."
    echo "移动 Qwen 目录失败。"
    exit 1
  fi
else
  echo "Directory Kedreamix/Linly-Talker/Qwen does not exist, cannot move Qwen model."
  echo "目录 Kedreamix/Linly-Talker/Qwen 不存在，无法移动 Qwen 模型。"
  exit 1
fi

# 6. MuseTalk模型
# MuseTalk model
mkdir -p ./Musetalk/models/
if [ -d "Kedreamix/Linly-Talker/MuseTalk" ]; then
  mv Kedreamix/Linly-Talker/MuseTalk/* ./Musetalk/models/
  if [ $? -ne 0 ]; then
    echo "Failed to move MuseTalk/models."
    echo "移动 MuseTalk/models 失败。"
    exit 1
  fi
else
  echo "Directory Kedreamix/Linly-Talker/MuseTalk does not exist, cannot move MuseTalk model."
  echo "目录 Kedreamix/Linly-Talker/MuseTalk 不存在，无法移动 MuseTalk 模型。"
  exit 1
fi

# 7. WhisperASR模型
# WhisperASR model
if [ -d "Kedreamix/Linly-Talker/Whisper" ]; then
  mv Kedreamix/Linly-Talker/Whisper ./
  if [ $? -ne 0 ]; then
    echo "Failed to move Whisper directory."
    echo "移动 Whisper 目录失败。"
    exit 1
  fi
else
  echo "Directory Kedreamix/Linly-Talker/Whisper does not exist, cannot move Whisper model."
  echo "目录 Kedreamix/Linly-Talker/Whisper 不存在，无法移动 Whisper 模型。"
  exit 1
fi

# 8. FunASR模型
# FunASR model
if [ -d "Kedreamix/Linly-Talker/FunASR" ]; then
  mv Kedreamix/Linly-Talker/FunASR ./
  if [ $? -ne 0 ]; then
    echo "Failed to move FunASR directory."
    echo "移动 FunASR 目录失败。"
    exit 1
  fi
else
  echo "Directory Kedreamix/Linly-Talker/FunASR does not exist, cannot move FunASR model."
  echo "目录 Kedreamix/Linly-Talker/FunASR 不存在，无法移动 FunASR 模型。"
  exit 1
fi

# 9. CosyVoice模型
# CosyVoice model
# Check if the CosyVoice checkpoints directory exists
if [ -d "checkpoints/CosyVoice_ckpt" ]; then
  # Create the CosyVoice/pretrained_models directory if it doesn't exist
  mkdir -p CosyVoice/pretrained_models
  
  # Move the CosyVoice-ttsfrd directory to the CosyVoice/pretrained_models directory
  mv checkpoints/CosyVoice_ckpt/CosyVoice-ttsfrd CosyVoice/pretrained_models
  
  # Check if the move operation was successful
  if [ $? -ne 0 ]; then
    echo "Failed to move CosyVoice-ttsfrd directory."
    echo "移动 CosyVoice-ttsfrd 目录失败。"
    exit 1
  fi

  # Unzip the resource.zip file inside the CosyVoice-ttsfrd directory
  unzip CosyVoice/pretrained_models/CosyVoice-ttsfrd/resource.zip -d CosyVoice/pretrained_models/CosyVoice-ttsfrd
  pip install CosyVoice/pretrained_models/CosyVoice-ttsfrd/ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
  
  # Check if the unzip operation was successful
  if [ $? -ne 0 ]; then
    echo "Failed to unzip resource.zip."
    echo "解压 resource.zip 失败。"
    exit 1
    
  fi
else
  echo "Directory Kedreamix/Linly-Talker/checkpoints/CosyVoice_ckpt does not exist, cannot move CosyVoice model."
  echo "目录 Kedreamix/Linly-Talker/checkpoints/CosyVoice_ckpt 不存在，无法移动 CosyVoice 模型。"
  exit 1
fi

echo "All models have been successfully moved and are ready."
echo "所有模型已成功移动并准备就绪。"
