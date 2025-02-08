from transformers import MarianMTModel, MarianTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
from fastapi import APIRouter, HTTPException, UploadFile, File
import torch
import torchaudio
import os

router = APIRouter()

# Translation model to handle speech to text

# Translation model setup (Helsinki-NLP/opus-mt-en-de)
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

# Speech recognition model setup (Wav2Vec2)
speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Enable FFmpeg for AAC support
torchaudio.set_audio_backend("ffmpeg")

@router.post("/translate")
async def translate(text: str):
    # Perform translation
    tokens = translation_tokenizer.encode(text, return_tensors="pt")
    translated_tokens = translation_model.generate(tokens, max_length=200)
    translated_text = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return {"translated_text": translated_text}

@router.post("/speech-to-text")
async def speech_to_text(audio_file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_ext = audio_file.filename.split(".")[-1]
        file_path = f"temp_audio.{file_ext}"

        with open(file_path, "wb") as f:
            f.write(await audio_file.read())

        # Load the audio file with FFmpeg backend (supports AAC)
        waveform, sample_rate = torchaudio.load(file_path)

        # Process the audio with Wav2Vec2
        inputs = speech_processor(waveform, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            logits = speech_model(input_values=inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = speech_processor.decode(predicted_ids[0])

        # Delete the temp file
        os.remove(file_path)

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
