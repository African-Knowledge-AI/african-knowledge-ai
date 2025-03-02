#from fastapi import APIRouter, Query, HTTPException
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
import io
import subprocess
import requests
from .bias_checker import check_bias
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
dotenv_path = os.path.join(os.getcwd(), ".env")
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

router = APIRouter()

# Securely load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Hugging Face API key

HF_WHISPER_MODEL = "openai/whisper-base"
HF_TTS_API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts"




# ✅ OpenAI Client
openai_client = openai.OpenAI()

# Supported African Languages
AFRICAN_LANGUAGES = {
    "eng": "English",
    "hau": "Hausa",
    "yor": "Yoruba",
    "ibo": "Igbo",
    "swa": "Swahili",
    "amh": "Amharic",
    "wol": "Wolof",
    "zul": "Zulu",
    "xho": "Xhosa",
    "sna": "Shona",
}

# ✅ Hugging Face Translation API Function

def translator(text: str, src_lang: str, tgt_lang: str):
    url = "https://api-inference.huggingface.co/models/facebook/m2m100_418M"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    payload = {
        "inputs": text, 
        "parameters": {},
        "options": {"use_cache": False}
    }
    
    headers["X-Model-Specific-Args"] = (
        f'{{"forced_bos_token": "<{tgt_lang}>", "source_lang": "{src_lang}"}}'
    )
    
    #response = requests.post(url, headers=headers, json=payload)
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    
    if response.status_code == 200:
        return response.json()[0]["translation_text"]
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Translation API error: {response.text}")

# ✅ Translation API Endpoint
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    
    

HF_WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-base"

# ✅ Speech-to-Text API
@router.post("/speech-to-text")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    model: str = Query("openai", enum=["openai", "huggingface"])
):
    """
    Transcribes speech to text using OpenAI Whisper or Hugging Face Whisper API.
    """
    try:
        file_ext = audio_file.filename.split(".")[-1].lower()
        original_path = f"temp_audio.{file_ext}"
        converted_path = "converted_audio.mp3"

        # Save uploaded file
        with open(original_path, "wb") as f:
            f.write(await audio_file.read())

        # Convert non-standard formats to MP3
        if file_ext not in ["flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"]:
            subprocess.run(["ffmpeg", "-i", original_path, "-ac", "1", "-ar", "16000", "-y", converted_path], check=True)
            file_to_use = converted_path
        else:
            file_to_use = original_path

        transcription = None
        model_used = None

        # ✅ OpenAI Whisper
        if model == "openai":
            with open(file_to_use, "rb") as audio:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.audio.transcriptions.create(model="whisper-1", file=audio)
                transcription = response.text
                model_used = "OpenAI Whisper"

        # ✅ Hugging Face Whisper API
        elif model == "huggingface":
            with open(file_to_use, "rb") as audio:
                headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
                response = requests.post(HF_WHISPER_API_URL, headers=headers, files={"file": audio})

            if response.status_code == 200:
                transcription = response.json().get("text", "Transcription failed.")
                model_used = "Hugging Face Whisper API"
            else:
                raise HTTPException(status_code=response.status_code, detail=f"Hugging Face API Error: {response.text}")

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



@router.post("/translate")
async def translate_text(
    request: TranslationRequest,
    model: str = Query(..., description="Specify 'openai' or 'huggingface'")
):
    """
    Translates text using OpenAI GPT or Hugging Face API.
    """
    if request.source_lang not in AFRICAN_LANGUAGES or request.target_lang not in AFRICAN_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language. Use a valid language code.")
    
    try:
        if model == "openai":
            prompt = (f"Translate the following text from {AFRICAN_LANGUAGES[request.source_lang]} "
                      f"to {AFRICAN_LANGUAGES[request.target_lang]}: {request.text}")
            
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            translated_text = response.choices[0].message.content
            model_used = "OpenAI GPT"
        
        elif model == "huggingface":
            translated_text = translator(request.text, src_lang=request.source_lang, tgt_lang=request.target_lang)
            model_used = "Hugging Face M2M-100"
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model selection. Choose 'openai' or 'huggingface'.")
        
        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "source_language": AFRICAN_LANGUAGES[request.source_lang],
            "target_language": AFRICAN_LANGUAGES[request.target_lang],
            "model_used": model_used
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ TEXT-TO-SPEECH (TTS)
class TextToSpeechRequest(BaseModel):
    text: str
    language: str


@router.post("/text-to-speech")
async def text_to_speech(
    request: TextToSpeechRequest,
    model: str = Query("openai", enum=["openai", "huggingface"])
):
    """
    Converts text to speech using OpenAI TTS or Hugging Face TTS.
    """
    if request.language not in AFRICAN_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language.")

    try:
        audio_io = io.BytesIO()  # Create in-memory buffer

        # ✅ OpenAI TTS
        if model == "openai":
            response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=request.text
            )
            audio_io.write(response.content)  # Write binary data to buffer

        # ✅ Hugging Face TTS
        elif model == "huggingface":
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            payload = {"inputs": request.text}
            response = requests.post(HF_TTS_API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                audio_io.write(response.content)  # Write binary data to buffer
            else:
                raise HTTPException(status_code=response.status_code, detail="TTS API error.")

        audio_io.seek(0)  # Reset buffer position

        # ✅ Return as streaming audio response
        return StreamingResponse(
            audio_io,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class ChatRequest(BaseModel):
    user_input: str

def get_african_response(user_input: str):
    """Fetch response from an external API instead of using transformers."""
    API_URL = "https://api-inference.huggingface.co/models/african-nlp/african-gpt2"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": user_input})
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    return "Error fetching African model response."
def get_response(user_input: str):
    """Handles query processing using ChatGPT and bias detection"""
    try:
        # Step 1: Get response from ChatGPT
        chatgpt_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_input}]
        ).choices[0].message.content

        # Step 2: Get response from African-aware model via API
        african_response = get_african_response(user_input)
        
                # Handle failure in fetching African model response
        if african_response.startswith("Error"):
            return {
                "original": chatgpt_response,
                "corrected": "Error fetching African model response.",
                "bias_score": "N/A",
                "explanation": "Bias score not available due to failure in retrieving the African model response."
            }


        # Step 3: Check bias in ChatGPT's response
        corrected_response, bias_score, explanation = check_bias(chatgpt_response, african_response)

        return {
            "original": chatgpt_response,
            "corrected": corrected_response,
            "bias_score": bias_score,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint for getting responses with bias detection."""
    return get_response(request.user_input)
