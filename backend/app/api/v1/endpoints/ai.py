from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os
import subprocess
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

# Load environment variables
load_dotenv()
dotenv_path = os.path.join(os.getcwd(), ".env")
print(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

 

router = APIRouter()

# Securely load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_WHISPER_MODEL = "openai/whisper-base"

# Load translation model
def load_translation_model():
    return pipeline("translation", model="facebook/nllb-200-distilled-600M")

translator = load_translation_model()

# ✅ Initialize OpenAI Client
openai_client = openai.OpenAI()

# Supported African Languages
AFRICAN_LANGUAGES = {
    "eng": "English",
    "hau": "Hausa",
    "yo": "Yoruba",
    "ibo": "Igbo",
    "swa": "Swahili",
    "amh": "Amharic",
    "wol": "Wolof",
    "zul": "Zulu",
    "xho": "Xhosa",
    "sna": "Shona",
}

# Request Model for Translation API
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@router.post("/translate")
async def translate_text(
    request: TranslationRequest,
    model: str = Query(..., description="Specify 'openai' or 'huggingface'")
):
    """
    Translates text using OpenAI GPT or Hugging Face NLLB.
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
            translation = translator(request.text, src_lang=request.source_lang, tgt_lang=request.target_lang)
            translated_text = translation[0]["translation_text"] if translation else "Translation failed"
            model_used = "Hugging Face NLLB"
        
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
