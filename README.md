---
title: cobytartil-audio-api
emoji: 🎙️
colorFrom: green
colorTo: teal
sdk: docker
pinned: false
---

# cobytartil-audio-api

Proxy API for Arabic speech transcription using OpenAI Whisper (via HuggingFace Inference API).

## Endpoint

`POST /transcribe` — accepts a multipart audio file, returns `{ "text": "..." }`.

## Environment

Set `HF_TOKEN` as a Space secret (HuggingFace → Settings → Secrets).
