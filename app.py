import mimetypes
import os
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient

# Python maps .webm → video/webm by default; override so HF accepts it
mimetypes.add_type("audio/webm", ".webm")

_EXT = {
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/wav": ".wav", "audio/wave": ".wav", "audio/x-wav": ".wav",
    "audio/webm": ".webm", "video/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/m4a": ".m4a", "audio/x-m4a": ".m4a",
}

HF_TOKEN = os.environ["HF_TOKEN"]
client = InferenceClient(token=HF_TOKEN)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    audio = await file.read()
    ext = _EXT.get(file.content_type or "", ".webm")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio)
        tmp_path = tmp.name
    try:
        result = client.automatic_speech_recognition(
            tmp_path, model="tarteel-ai/whisper-base-ar-quran"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        os.unlink(tmp_path)
    return {"text": result.text}
