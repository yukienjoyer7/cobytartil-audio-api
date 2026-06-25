import base64
import json
import os
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ["HF_TOKEN"]
hf = InferenceClient(token=HF_TOKEN)

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
    try:
        raw = hf.post(
            json={
                "inputs": base64.b64encode(audio).decode(),
                "parameters": {"language": "arabic", "task": "transcribe"},
            },
            model="openai/whisper-large-v3",
            task="automatic-speech-recognition",
        )
        result = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return result
