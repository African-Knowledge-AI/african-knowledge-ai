from transformers import MarianMTModel, MarianTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torchaudio

router = APIRouter()

# Translation model setup (Helsinki-NLP/opus-mt-en-de)
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

# Speech recognition model setup (Wav2Vec2)
speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

class TranslationRequest(BaseModel):
    text: str

class SpeechRequest(BaseModel):
    audio_file_path: str  # Path to the audio file

@router.post("/translate")
async def translate(request: TranslationRequest):
    # Perform translation
    tokens = translation_tokenizer.encode(request.text, return_tensors="pt")
    translated_tokens = translation_model.generate(tokens, max_length=200)
    translated_text = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return {"translated_text": translated_text}

@router.post("/speech-to-text")
async def speech_to_text(request: SpeechRequest):
    # Load the audio file and process
    waveform, sample_rate = torchaudio.load(request.audio_file_path)
    inputs = speech_processor(waveform, return_tensors="pt", sampling_rate=sample_rate)
    with torch.no_grad():
        logits = speech_model(input_values=inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = speech_processor.decode(predicted_ids[0])
    return {"transcription": transcription}
