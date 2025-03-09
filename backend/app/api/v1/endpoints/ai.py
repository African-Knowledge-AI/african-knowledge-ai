from fastapi import APIRouter, UploadFile, File, Query, HTTPException
import re
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import openai
import os
import io
import json
import subprocess
import requests
from .bias_checker import check_bias
from dotenv import load_dotenv
from typing import Dict, List, Union 

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
    
    
    # Define input model
class ChatRequest(BaseModel):
    user_input: str

def get_african_response(text: str) -> Union[List[Dict[str, float]], str]:
    """Fetch bias detection response from Hugging Face API."""
    API_URL = "https://api-inference.huggingface.co/models/d4data/bias-detection-model"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        logging.info(f"African Model API Response Status: {response.status_code}")

        if response.status_code == 200:
            response_json = response.json()
            if isinstance(response_json, list) and response_json:
                return response_json[0] if isinstance(response_json[0], list) else []
        return "Error fetching African model response."
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Exception: {e}")
        return "Error: Failed to connect to the African model API."

def classify_bias_level(score: float) -> str:
    """Classifies bias on a nuanced scale."""
    if score < 0.3:
        return "Low"
    elif score < 0.7:
        return "Moderate"
    return "High"

def flag_biased_keywords(text: str) -> Dict[str, List[str]]:
    """Identifies biased keywords and categorizes them."""
    bias_keywords = {
        "racial": ["uncivilized", "savage", "primitive"],
        "economic": ["poor", "underdeveloped", "third-world"],
        "political": ["corrupt", "dictatorship", "failed state"]
    }

    flagged = {}
    for category, keywords in bias_keywords.items():
        matches = [word for word in keywords if re.search(rf"\b{word}\b", text, re.IGNORECASE)]
        if matches:
            flagged[category] = matches

    return flagged

def neutralize_text(original_text: str) -> str:
    """Uses OpenAI GPT to generate a neutral, fact-based version of a given text."""
    prompt = f"""
    The following text may contain bias. Please rewrite it in a neutral, fact-based manner:
    
    Original: "{original_text}"
    
    Neutral version:
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Create client
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        logging.error(f"Error correcting text: {e}")
        return f"Error: {str(e)}"

def fact_check_text(text: str) -> Dict[str, Dict[str, str]]:
    """Uses OpenAI GPT-4 to fact-check text and return verification results along with correct information."""
    prompt = f"""
    You are a fact-checking assistant. Analyze the following text and determine if each statement is true, false, or needs verification. 
    
    Text: "{text}"
    
    For each statement in the text, provide:
    - "verification" (True, False, or Needs Verification)
    - "correction" (A corrected statement if false, or additional context if needed)

    Return your response strictly in JSON format:
    {{
        "statement_1": {{"verification": "True/False/Needs Verification", "correction": "Corrected statement or additional context"}},
        "statement_2": {{"verification": "True/False/Needs Verification", "correction": "Corrected statement or additional context"}}
    }}
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        raw_text = response.choices[0].message.content.strip()

        # Extract valid JSON response
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))  # Extract and parse JSON

        logging.error("Error: OpenAI response is not valid JSON")
        return {"error": "Invalid JSON response from OpenAI"}

    except openai.OpenAIError as e:
        logging.error(f"Error in fact-checking: {e}")
        return {"error": "Failed to connect to OpenAI API."}
    except json.JSONDecodeError:
        logging.error("JSON parsing failed. OpenAI response was not properly formatted.")
        return {"error": "Invalid JSON response from OpenAI"}

def get_response(user_input: str) -> Dict[str, Union[str, float, bool, Dict]]:
    """Processes user input, detects bias, fact-checks, and suggests corrections."""
    try:
        # Step 1: Fetch bias analysis
        african_response = get_african_response(user_input)
        if isinstance(african_response, str) and "Error" in african_response:
            return {
                "original": user_input,
                "corrected": "Error fetching bias detection results.",
                "bias_score": "N/A",
                "bias_level": "Unknown",
                "flagged_keywords": {},
                "fact_check_results": "N/A",
                "toggle_correction": False,
                "explanation": "Bias score unavailable. Manual review recommended."
            }

        # Step 2: Determine highest bias score
        bias_score = max((entry.get("score", 0.0) for entry in african_response), default=0.0)
        bias_label = classify_bias_level(bias_score)

        # Step 3: Flag biased keywords manually
        flagged_keywords = flag_biased_keywords(user_input)

        # Step 4: Apply correction logic
        if bias_label == "Low":
            corrected_text = user_input
            toggle_correction = False
            explanation = "Bias level is low. No significant modifications were necessary."
        else:  # Moderate or High Bias → Rewrite text
            corrected_text = neutralize_text(user_input)
            toggle_correction = True
            explanation = f"{bias_label} bias detected (Score: {round(bias_score, 2)}). A neutral version has been generated, but review is recommended."

        # Step 5: Fact-checking
        fact_check_results = fact_check_text(user_input)

        return {
            "original": user_input,
            "corrected": corrected_text,
            "bias_score": round(bias_score, 2),
            "bias_level": bias_label,
            "flagged_keywords": flagged_keywords,
            "fact_check_results": fact_check_results,  # Now properly formatted JSON
            "toggle_correction": toggle_correction,
            "explanation": explanation
        }

    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint
from fastapi import APIRouter

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint for chat responses with bias detection and fact-checking."""
    return get_response(request.user_input)
    
    
'''     # Define input model
class ChatRequest(BaseModel):
    user_input: str


def get_african_response(text: str) -> Union[List[Dict[str, float]], str]:
    """Fetch bias detection response from Hugging Face API."""
    API_URL = "https://api-inference.huggingface.co/models/d4data/bias-detection-model"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        logging.info(f"African Model API Response Status: {response.status_code}")

        if response.status_code == 200:
            response_json = response.json()
            if isinstance(response_json, list) and response_json:
                return response_json[0] if isinstance(response_json[0], list) else []
        return "Error fetching African model response."
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Exception: {e}")
        return "Error: Failed to connect to the African model API."

def classify_bias_level(score: float) -> str:
    """Classifies bias on a nuanced scale."""
    if score < 0.3:
        return "Low"
    elif score < 0.7:
        return "Moderate"
    return "High"

def flag_biased_keywords(text: str) -> Dict[str, List[str]]:
    """Identifies biased keywords and categorizes them."""
    bias_keywords = {
        "racial": ["uncivilized", "savage", "primitive"],
        "economic": ["poor", "underdeveloped", "third-world"],
        "political": ["corrupt", "dictatorship", "failed state"]
    }

    flagged = {}
    for category, keywords in bias_keywords.items():
        matches = [word for word in keywords if re.search(rf"\b{word}\b", text, re.IGNORECASE)]
        if matches:
            flagged[category] = matches

    return flagged

def neutralize_text(original_text: str) -> str:
    """Uses OpenAI GPT to generate a neutral, fact-based version of a given text."""
    prompt = f"""
    The following text may contain bias. Please rewrite it in a neutral, fact-based manner:
    
    Original: "{original_text}"
    
    Neutral version:
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Create client
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        logging.error(f"Error correcting text: {e}")
        return f"Error: {str(e)}"

def fact_check_text(text: str) -> Dict[str, str]:
    """Uses OpenAI GPT-4 to fact-check the text and return a verification result."""
    prompt = f"""
    You are a fact-checking assistant. Analyze the following text and determine if each statement is true, false, or needs verification. 
    
    Text: "{text}"
    
    Return your response in JSON format:
    {{
        "statement_1": "True/False/Needs Verification",
        "statement_2": "True/False/Needs Verification",
        ...
    }}
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        logging.error(f"Error in fact-checking: {e}")
        return {"error": "Failed to fact-check the text."}

def get_response(user_input: str) -> Dict[str, Union[str, float, bool, Dict]]:
    """Processes user input, detects bias, fact-checks, and suggests corrections."""
    try:
        # Step 1: Fetch bias analysis
        african_response = get_african_response(user_input)
        if isinstance(african_response, str) and "Error" in african_response:
            return {
                "original": user_input,
                "corrected": "Error fetching bias detection results.",
                "bias_score": "N/A",
                "bias_level": "Unknown",
                "flagged_keywords": {},
                "fact_check_results": "N/A",
                "toggle_correction": False,
                "explanation": "Bias score unavailable. Manual review recommended."
            }

        # Step 2: Determine highest bias score
        bias_score = max((entry.get("score", 0.0) for entry in african_response), default=0.0)
        bias_label = classify_bias_level(bias_score)

        # Step 3: Flag biased keywords manually
        flagged_keywords = flag_biased_keywords(user_input)

        # Step 4: Apply correction logic
        if bias_label == "Low":
            corrected_text = user_input
            toggle_correction = False
            explanation = "Bias level is low. No significant modifications were necessary."
        else:  # Moderate or High Bias → Rewrite text
            corrected_text = neutralize_text(user_input)
            toggle_correction = True
            explanation = f"{bias_label} bias detected (Score: {round(bias_score, 2)}). A neutral version has been generated, but review is recommended."

        # Step 5: Fact-checking
        fact_check_results = fact_check_text(user_input)

        return {
            "original": user_input,
            "corrected": corrected_text,
            "bias_score": round(bias_score, 2),
            "bias_level": bias_label,
            "flagged_keywords": flagged_keywords,
            "fact_check_results": fact_check_results,
            "toggle_correction": toggle_correction,
            "explanation": explanation
        }

    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(request: ChatRequest):
    """API endpoint for chat responses with bias detection and fact-checking."""
    return get_response(request.user_input)
    
  '''