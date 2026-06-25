import os
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient

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
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio)
        tmp_path = tmp.name
    try:
        result = client.automatic_speech_recognition(
            tmp_path, model="openai/whisper-large-v3-turbo"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        os.unlink(tmp_path)
    return {"text": result.text}
