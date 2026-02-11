"""
Linly-Talker Educational UI
This interface helps you understand how the digital human conversation system works.
"""
import gradio as gr
import os

def get_architecture_diagram():
    """Return the system architecture explanation"""
    return """
# Linly-Talker System Architecture

## Overview
Linly-Talker is a Digital Human Intelligent Dialogue System that creates interactive AI-powered conversations with realistic avatars.

## System Flow

```
┌─────────────┐
│   Input     │
│ (Voice/Text)│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  1. ASR (Automatic Speech Recognition)  │
│  - Converts voice to text               │
│  - Models: Whisper, FunASR              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2. LLM (Large Language Model)          │
│  - Generates intelligent responses      │
│  - Models: Linly, Qwen, ChatGPT, etc.   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3. TTS (Text-to-Speech)                │
│  - Converts text to speech              │
│  - Models: Edge-TTS, GPT-SoVITS         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  4. THG (Talking Head Generation)       │
│  - Animates avatar with speech          │
│  - Models: SadTalker, Wav2Lip, MuseTalk │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────┐
│  Output Video   │
│ (Digital Human) │
└─────────────────┘
```
"""

def get_component_details(component):
    """Return detailed information about each component"""
    details = {
        "ASR": """
## ASR - Automatic Speech Recognition

**Purpose:** Convert spoken audio into text

**How it works:**
1. User speaks into microphone or uploads audio file
2. ASR model processes audio waveform
3. Converts speech to text transcription
4. Output text is sent to LLM

**Available Models:**
- **Whisper** (OpenAI): Accurate multilingual speech recognition
- **FunASR** (Alibaba): Fast Chinese speech recognition
- **OmniSenseVoice**: Fast speech recognition with good accuracy

**Example:**
- Input: [Audio: "你好，今天天气怎么样？"]
- Output: "你好，今天天气怎么样？"
        """,

        "LLM": """
## LLM - Large Language Model

**Purpose:** Generate intelligent conversational responses

**How it works:**
1. Receives text from ASR (or direct text input)
2. Processes the question/statement using AI
3. Generates contextually appropriate response
4. Maintains conversation history for multi-turn dialogue

**Available Models:**
- **Linly-AI**: Chinese language model
- **Qwen** (Alibaba): Efficient Chinese & English model
- **ChatGPT** (OpenAI): Advanced conversational AI
- **Gemini-Pro** (Google): Multimodal AI model
- **ChatGLM** (Tsinghua): Chinese conversational model

**Example:**
- Input: "今天天气怎么样？"
- LLM Processing: [Context + Question Analysis]
- Output: "今天天气晴朗，温度适中，是个出门的好日子。"
        """,

        "TTS": """
## TTS - Text-to-Speech

**Purpose:** Convert text responses into natural-sounding speech

**How it works:**
1. Receives text from LLM
2. Converts text to audio waveform
3. Can clone specific voices
4. Generates expressive speech with proper intonation

**Available Models:**
- **Edge-TTS** (Microsoft): Free online TTS with multiple voices
- **PaddleTTS**: Offline Chinese TTS
- **GPT-SoVITS**: Advanced voice cloning (clone voice from 1-minute sample)
- **CosyVoice** (Alibaba): High-quality multilingual TTS
- **XTTS**: Cross-lingual voice cloning

**Features:**
- Voice cloning: Copy anyone's voice
- Emotion control: Happy, sad, angry, etc.
- Multi-language support: Chinese, English, Japanese, Korean

**Example:**
- Input: "今天天气晴朗，温度适中，是个出门的好日子。"
- Output: [Audio file with natural speech]
        """,

        "THG": """
## THG - Talking Head Generation

**Purpose:** Create realistic animated avatars that sync with speech

**How it works:**
1. Takes input image (digital human photo)
2. Receives audio from TTS
3. Animates face to match speech
4. Generates realistic lip-sync and expressions
5. Outputs video of talking digital human

**Available Models:**
- **SadTalker** (CVPR 2023): High-quality facial animation
- **Wav2Lip**: Accurate lip synchronization
- **Wav2Lipv2**: Enhanced version with better quality
- **ER-NeRF** (ICCV 2023): NeRF-based realistic rendering
- **MuseTalk**: Real-time conversation (30+ FPS)

**Features:**
- Upload any photo to create digital human
- Realistic facial expressions
- Natural head movements
- Real-time processing (MuseTalk)

**Example:**
- Input: Photo + Audio
- Output: Video of person speaking the audio content
        """
    }
    return details.get(component, "Component not found")

def get_use_cases():
    """Return practical use cases"""
    return """
## Practical Use Cases

### 1. Customer Service
- 24/7 AI receptionist
- Answer common questions
- Natural voice interaction

### 2. Education
- Virtual teacher
- Interactive learning
- Personalized tutoring

### 3. Content Creation
- Video narration
- News anchor
- Social media content

### 4. Personal Assistant
- Voice-controlled helper
- Schedule management
- Information retrieval

### 5. Entertainment
- Virtual companions
- Interactive storytelling
- Gaming characters

### 6. Accessibility
- Voice interface for visually impaired
- Text-to-speech for reading
- Multilingual support
    """

def get_quick_start():
    """Return quick start guide"""
    return """
## Quick Start Guide

### Running the WebUI

```bash
# 1. Activate environment
conda activate linly

# 2. Run the main WebUI
python webui.py

# 3. Open browser
# Navigate to: http://localhost:7870
```

### Basic Usage

1. **Choose Your Mode:**
   - Text conversation
   - Voice conversation
   - Upload custom images

2. **Select Models:**
   - ASR Model: Whisper or FunASR
   - LLM Model: Qwen, ChatGPT, or others
   - TTS Model: Edge-TTS or GPT-SoVITS
   - THG Model: SadTalker, Wav2Lip, or MuseTalk

3. **Start Conversation:**
   - Type text or speak
   - AI generates response
   - Watch digital human speak

### File Structure

```
Linly-Talker/
├── ASR/          # Speech recognition models
├── LLM/          # Language models
├── TTS/          # Text-to-speech models
├── TFG/          # Talking head generation
├── VITS/         # Voice cloning
├── checkpoints/  # Model weights
├── webui.py      # Main interface
└── configs.py    # Configuration
```
    """

def get_model_comparison():
    """Return model comparison table"""
    return """
## Model Comparison

### ASR Models
| Model | Speed | Accuracy | Language | Use Case |
|-------|-------|----------|----------|----------|
| Whisper | Medium | High | Multi-lang | General purpose |
| FunASR | Fast | High (Chinese) | Chinese | Chinese speech |
| OmniSenseVoice | Very Fast | High | Multi-lang | Real-time |

### LLM Models
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Qwen-1.8B | Small | Fast | Good | Resource-limited |
| Linly | Medium | Medium | Good | Chinese focus |
| ChatGPT | Large | Fast (API) | Excellent | Best quality |
| Gemini-Pro | Large | Fast (API) | Excellent | Multimodal |

### TTS Models
| Model | Quality | Speed | Voice Clone | Use Case |
|-------|---------|-------|-------------|----------|
| Edge-TTS | Good | Fast | No | Quick testing |
| GPT-SoVITS | Excellent | Medium | Yes | Voice cloning |
| CosyVoice | Excellent | Fast | Yes | Production |
| PaddleTTS | Good | Fast | No | Offline |

### THG Models
| Model | Quality | Speed | FPS | Use Case |
|-------|---------|-------|-----|----------|
| SadTalker | Excellent | Medium | 15-20 | High quality |
| Wav2Lip | Good | Fast | 20-25 | Lip-sync focus |
| Wav2Lipv2 | Very Good | Fast | 20-25 | Enhanced quality |
| MuseTalk | Good | Very Fast | 30+ | Real-time chat |
| ER-NeRF | Excellent | Slow | 10-15 | Highest quality |
    """

def get_technical_details():
    """Return technical implementation details"""
    return """
## Technical Implementation

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for all models
- **OS**: Linux, Windows, macOS

### Dependencies
```python
# Core frameworks
torch >= 2.0.0
gradio >= 4.16.0
transformers
torchaudio

# Specialized libraries
whisper (ASR)
edge-tts (TTS)
opencv-python (Video processing)
```

### Pipeline Execution

**1. Input Processing**
```python
# Voice input
audio = record_audio()
text = whisper.transcribe(audio)

# Or text input
text = "你好，今天天气怎么样？"
```

**2. LLM Response**
```python
# Generate response
response = llm.generate(
    prompt=text,
    history=conversation_history,
    max_length=100
)
```

**3. Speech Synthesis**
```python
# Convert to speech
audio = tts.synthesize(
    text=response,
    voice="zh-CN-XiaoxiaoNeural",
    rate="+0%",
    volume="+0%"
)
```

**4. Video Generation**
```python
# Animate digital human
video = thg.generate(
    source_image="avatar.jpg",
    driven_audio=audio,
    still_mode=False
)
```

### Performance Optimization

**Memory Management:**
- Automatic GPU cache clearing
- Lazy model loading
- Batch processing

**Speed Improvements:**
- Model quantization
- TensorRT optimization
- Streaming inference

**Quality Enhancements:**
- Face enhancement (GFPGAN)
- Super resolution
- Expression control
    """

# Create Gradio interface
with gr.Blocks(title="Understanding Linly-Talker", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 Understanding Linly-Talker
    ## Interactive Educational Guide to Digital Human Conversation System

    This interface helps you understand how Linly-Talker works, its components, and how to use it.
    """)

    with gr.Tabs():
        # Tab 1: System Architecture
        with gr.Tab("📊 System Architecture"):
            gr.Markdown(get_architecture_diagram())
            gr.Image("docs/HOI_en.png", label="System Architecture Diagram", show_label=True)

        # Tab 2: Components Deep Dive
        with gr.Tab("🔧 Components"):
            component_selector = gr.Radio(
                choices=["ASR", "LLM", "TTS", "THG"],
                label="Select Component to Learn About",
                value="ASR"
            )
            component_details = gr.Markdown(get_component_details("ASR"))

            component_selector.change(
                fn=get_component_details,
                inputs=component_selector,
                outputs=component_details
            )

        # Tab 3: Model Comparison
        with gr.Tab("⚖️ Model Comparison"):
            gr.Markdown(get_model_comparison())

        # Tab 4: Use Cases
        with gr.Tab("💡 Use Cases"):
            gr.Markdown(get_use_cases())

        # Tab 5: Quick Start
        with gr.Tab("🚀 Quick Start"):
            gr.Markdown(get_quick_start())

        # Tab 6: Technical Details
        with gr.Tab("⚙️ Technical Details"):
            gr.Markdown(get_technical_details())

        # Tab 7: Interactive Demo Flow
        with gr.Tab("🎮 Interactive Flow"):
            gr.Markdown("""
            ## Interactive Pipeline Demonstration

            Follow along to see how data flows through the system:
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Input")
                    demo_input = gr.Textbox(
                        label="User Input (Text/Voice)",
                        value="你好，今天天气怎么样？",
                        lines=3
                    )
                    gr.Markdown("""
                    **Source:** User speaks or types a question
                    """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 2: ASR Processing")
                    asr_output = gr.Textbox(
                        label="ASR Output (Text)",
                        value="你好，今天天气怎么样？",
                        lines=3
                    )
                    gr.Markdown("""
                    **Process:** Audio → Text conversion
                    **Model:** Whisper/FunASR
                    """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 3: LLM Response")
                    llm_output = gr.Textbox(
                        label="LLM Generated Response",
                        value="今天天气晴朗，温度适中，大约23度，是个出门的好日子。建议您外出时带上太阳镜。",
                        lines=4
                    )
                    gr.Markdown("""
                    **Process:** Understanding + Generation
                    **Model:** Qwen/ChatGPT/Linly
                    """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 4: TTS Synthesis")
                    tts_output = gr.Textbox(
                        label="TTS Output",
                        value="[Audio file generated: answer.wav]",
                        lines=2
                    )
                    gr.Markdown("""
                    **Process:** Text → Speech conversion
                    **Model:** Edge-TTS/GPT-SoVITS
                    """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 5: THG Animation")
                    thg_output = gr.Textbox(
                        label="THG Output",
                        value="[Video file generated: output.mp4 - Digital human speaking]",
                        lines=2
                    )
                    gr.Markdown("""
                    **Process:** Photo + Audio → Talking video
                    **Model:** SadTalker/Wav2Lip/MuseTalk
                    """)

            gr.Markdown("""
            ### 🎯 Final Result
            A video showing a digital human (avatar) speaking the AI-generated response with:
            - ✅ Natural lip synchronization
            - ✅ Realistic facial expressions
            - ✅ Natural head movements
            - ✅ High-quality audio
            """)

        # Tab 8: FAQs
        with gr.Tab("❓ FAQs"):
            gr.Markdown("""
            ## Frequently Asked Questions

            ### Q1: What makes Linly-Talker special?
            **A:** Linly-Talker integrates 4 major AI technologies (ASR, LLM, TTS, THG) into one seamless pipeline,
            allowing you to create realistic talking digital humans with just a photo and text/voice input.

            ### Q2: Do I need expensive hardware?
            **A:** A GPU with 8GB VRAM is recommended, but you can use smaller models (like Qwen-1.8B)
            or cloud-based models (ChatGPT API) for lower resource requirements.

            ### Q3: Can I use my own voice and face?
            **A:** Yes! You can:
            - Upload any photo to create a digital human
            - Clone your voice using GPT-SoVITS (needs 1-minute audio sample)
            - Use the system with your own language models

            ### Q4: Which model should I choose?
            **A:** It depends on your use case:
            - **Fast prototyping:** Edge-TTS + Qwen + Wav2Lip
            - **Best quality:** GPT-SoVITS + ChatGPT + SadTalker
            - **Real-time chat:** FunASR + Qwen + MuseTalk
            - **Voice cloning:** GPT-SoVITS/CosyVoice + Any LLM + Any THG

            ### Q5: Is it free to use?
            **A:** The software is open-source (MIT License). However:
            - Local models: Free but need GPU
            - Cloud APIs (ChatGPT, Gemini): Require API keys and usage fees
            - Edge-TTS: Free but requires internet

            ### Q6: How do I improve the quality?
            **A:**
            1. Use higher quality input images (clear, front-facing)
            2. Enable face enhancement (GFPGAN)
            3. Choose better models (SadTalker > Wav2Lip for quality)
            4. Use voice cloning for more natural speech
            5. Adjust parameters (expression weight, still mode, etc.)

            ### Q7: Can it work offline?
            **A:** Yes, using:
            - Whisper (ASR)
            - Local LLM (Qwen, Linly)
            - PaddleTTS (TTS)
            - Any THG model (all local)

            ### Q8: What languages are supported?
            **A:**
            - Chinese, English (all models)
            - Japanese, Korean, Cantonese (CosyVoice)
            - 100+ languages (Whisper ASR)

            ### Q9: How long does it take to generate a video?
            **A:**
            - SadTalker: ~30-60 seconds for 10-second video
            - Wav2Lip: ~20-30 seconds for 10-second video
            - MuseTalk: Real-time (30+ FPS)

            ### Q10: Where can I get help?
            **A:**
            - GitHub Issues: https://github.com/Kedreamix/Linly-Talker/issues
            - Documentation: See README.md
            - Video tutorials: https://space.bilibili.com/241286257
            """)

    gr.Markdown("""
    ---
    ### 🔗 Useful Links
    - [GitHub Repository](https://github.com/Kedreamix/Linly-Talker)
    - [Video Tutorials (Bilibili)](https://space.bilibili.com/241286257/channel/collectiondetail?sid=2065753)
    - [Documentation](https://github.com/Kedreamix/Linly-Talker/blob/main/README.md)

    ### 📝 Next Steps
    1. ✅ Understand the architecture (you're here!)
    2. 🔧 Install dependencies: `pip install -r requirements_webui.txt`
    3. 📦 Download models: `sh scripts/download_models.sh`
    4. 🚀 Run WebUI: `python webui.py`
    5. 🎨 Experiment with different models and settings

    ---
    Made with ❤️ to help you understand Linly-Talker
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
