import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from parent camb-ai-work directory as fallback
_parent_env = Path(__file__).resolve().parents[2] / ".env"
if _parent_env.exists():
    load_dotenv(_parent_env, override=False)

sys.path.append("../Linly-Talker")
from src.cost_time import calculate_time

os.environ.setdefault("GRADIO_TEMP_DIR", "./temp")


class CambTTS:
    def __init__(
        self,
        api_key=None,
        voice_id=147320,
        language="en-us",
        speech_model="mars-flash",
    ):
        self.api_key = api_key or os.getenv("CAMB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "CAMB API key is required. Set CAMB_API_KEY env var or pass api_key."
            )
        self.voice_id = voice_id
        self.language = language
        self.speech_model = speech_model

        from camb.client import CambAI

        self.client = CambAI(api_key=self.api_key)

    @calculate_time
    def predict(
        self,
        TEXT,
        VOICE_ID=None,
        RATE=1.0,
        VOLUME=1.0,
        PITCH=1.0,
        OUTPUT_FILE="result.wav",
        OUTPUT_SUBS="result.vtt",
        words_in_cue=8,
    ):
        from camb.types import StreamTtsOutputConfiguration

        voice_id = int(VOICE_ID) if VOICE_ID else self.voice_id

        stream = self.client.text_to_speech.tts(
            text=TEXT,
            language=self.language,
            voice_id=voice_id,
            speech_model=self.speech_model,
            output_configuration=StreamTtsOutputConfiguration(format="wav"),
        )

        with open(OUTPUT_FILE, "wb") as f:
            for chunk in stream:
                f.write(chunk)

        return OUTPUT_FILE, None

    def clone_voice(self, audio_path, voice_name=None, gender=1, language=None):
        """Clone a voice from reference audio (10-30s of clear speech).

        Args:
            audio_path: Path to reference audio file (WAV, MP3, FLAC, OGG)
            voice_name: Name for the cloned voice (auto-generated if None)
            gender: 1 = male, 2 = female
            language: Language enum, defaults to self.language

        Returns:
            int: The voice_id of the cloned voice
        """
        from camb.types.language_enums import Languages

        lang_map = {
            "en-us": Languages.EN_US,
            "es-es": Languages.ES_ES,
            "fr-fr": Languages.FR_FR,
            "de-de": Languages.DE_DE,
            "ja-jp": Languages.JA_JP,
            "hi-in": Languages.HI_IN,
            "pt-br": Languages.PT_BR,
            "zh-cn": Languages.ZH_CN,
        }
        lang_enum = lang_map.get(language or self.language, Languages.EN_US)

        result = self.client.voice_cloning.create_custom_voice(
            file=open(audio_path, "rb"),
            voice_name=voice_name or f"camb_clone_{os.path.basename(audio_path)}",
            gender=gender,
            language=lang_enum,
            enhance_audio=True,
        )
        return result.voice_id

    @calculate_time
    def predict_translated(
        self,
        TEXT,
        SOURCE_LANG_ID=1,
        TARGET_LANG_ID=54,
        VOICE_ID=None,
        OUTPUT_FILE="translated_result.wav",
    ):
        """Translate text and synthesize speech in the target language.

        Args:
            TEXT: Text to translate and speak
            SOURCE_LANG_ID: Numeric source language ID (1=English, see CAMB docs)
            TARGET_LANG_ID: Numeric target language ID (54=Spanish, see CAMB docs)
            VOICE_ID: Voice ID for the target language
            OUTPUT_FILE: Output audio file path

        Returns:
            tuple: (output_file_path, None)
        """
        import time
        import requests

        voice_id = int(VOICE_ID) if VOICE_ID else self.voice_id

        response = self.client.translated_tts.create_translated_tts(
            text=TEXT,
            source_language=SOURCE_LANG_ID,
            target_language=TARGET_LANG_ID,
            voice_id=voice_id,
        )
        task_id = response.task_id

        # Poll for completion
        for _ in range(60):
            status = self.client.translated_tts.get_translated_tts_task_status(
                task_id=task_id
            )
            if status.status == "SUCCESS":
                result = self.client.text_to_speech.get_tts_run_info(
                    run_id=status.run_id,
                    output_type="file_url",
                )
                audio_response = requests.get(result.output_url)
                with open(OUTPUT_FILE, "wb") as f:
                    f.write(audio_response.content)
                return OUTPUT_FILE, None
            elif status.status in ("ERROR", "FAILED"):
                raise RuntimeError(f"Translated TTS failed: {status}")
            time.sleep(2)

        raise TimeoutError("Translated TTS task timed out")

    def list_voices(self):
        """List all available voices.

        Returns:
            list: List of voice dictionaries with id, voice_name, gender, etc.
        """
        return self.client.voice_cloning.list_voices()


def test():
    tts = CambTTS()
    TEXT = "Hello, I am CAMB AI. Nice to meet you!"
    audio_file, _ = tts.predict(TEXT, OUTPUT_FILE="camb_tts_test.wav")
    print("Audio file written to", audio_file)


if __name__ == "__main__":
    test()
