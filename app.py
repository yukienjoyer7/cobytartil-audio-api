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
        result = hf.automatic_speech_recognition(
            audio,
            model="openai/whisper-large-v3",
            extra_body={"language": "arabic", "task": "transcribe"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"text": result.text}
