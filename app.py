import os
import httpx
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

HF_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HF_TOKEN = os.environ["HF_TOKEN"]

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
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            HF_URL,
            content=audio,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": file.content_type or "audio/webm",
            },
        )
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()
