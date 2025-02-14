from fastapi import APIRouter, UploadFile, File, Query, HTTPException
import openai
from dotenv import load_dotenv
import os
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

load_dotenv()  # Load variables from .env
dotenv_path = os.path.join(os.getcwd(), ".env")
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY:", api_key)

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Securely load API key
HF_WHISPER_MODEL = "openai/whisper-base"  # Replace with your HF model name

@router.post("/speech-to-text")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    model: str = Query("openai", enum=["openai", "huggingface"])  # Accept `model` as a query parameter
):
    """
    Transcribes speech to text using either OpenAI Whisper or Hugging Face Whisper.
    
    Parameters:
    - `audio_file`: The uploaded audio file.
    - `model`: Choose between `"openai"` (default) or `"huggingface"`.
    
    Returns:
    - A JSON response with the transcription and model used.
    """
    try:
        file_ext = audio_file.filename.split(".")[-1].lower()
        original_path = f"temp_audio.{file_ext}"
        converted_path = "converted_audio.mp3"

        # Save the uploaded audio file
        with open(original_path, "wb") as f:
            f.write(await audio_file.read())

        # Convert if necessary
        if file_ext not in ["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"]:
            subprocess.run(["ffmpeg", "-i", original_path, "-ac", "1", "-ar", "16000", "-y", converted_path], check=True)
            file_to_use = converted_path
        else:
            file_to_use = original_path

        # OpenAI Whisper
        if model == "openai":
            with open(file_to_use, "rb") as audio:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.audio.transcriptions.create(model="whisper-1", file=audio)
                transcription = response.text
                model_used = "OpenAI Whisper"

        # Hugging Face Whisper
        elif model == "huggingface":
            speech_recognizer = pipeline("automatic-speech-recognition", model=HF_WHISPER_MODEL)
            transcription = speech_recognizer(file_to_use)["text"]
            model_used = "Hugging Face Whisper"

        else:
            raise HTTPException(status_code=400, detail="Invalid model selection. Choose 'openai' or 'huggingface'.")

        # Cleanup temporary files
        os.remove(original_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)

        return {
            "message": "File transcribed successfully",
            "transcription": transcription,
            "model_used": model_used
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





