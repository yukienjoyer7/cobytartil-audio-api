import os
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

_model = WhisperModel("base", device="cpu", compute_type="int8")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    audio = await file.read()
    suffix = ".webm" if (file.content_type or "").startswith("video") else ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio)
        tmp_path = tmp.name
    try:
        segments, _ = _model.transcribe(tmp_path, language="ar", task="transcribe")
        text = "".join(s.text for s in segments).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
    return {"text": text}
