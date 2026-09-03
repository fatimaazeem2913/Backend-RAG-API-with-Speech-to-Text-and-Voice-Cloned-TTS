# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# ffmpeg: required by faster-whisper and TTS for audio decoding.
# build-essential + git: some ML packages (chromadb, sentence-transformers
# deps) need to compile or fetch from git during pip install.
# libsndfile1: required by the `soundfile` package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first, separately from app code, so this expensive
# layer is only rebuilt when the lock file actually changes — not on
# every source code edit.
#
# 🚀 THE FIX: uses requirements-docker.txt, generated from `pip freeze`
# on the actual working local venv (then stripped of CUDA-only packages).
# The original loose requirements.txt (chromadb>=0.4.24 with no ceiling,
# etc.) let pip's resolver backtrack through an entire year of chromadb
# releases trying to find a compatible combination — every version is
# exactly pinned here instead, so there's nothing left to resolve.
COPY requirements-docker.txt .

# 🚀 install CPU-only torch/torchaudio FIRST, from PyTorch's dedicated
# CPU wheel index. Without this, pip resolves the default CUDA-enabled
# torch wheel — which drags in several GB of NVIDIA CUDA runtime packages
# (nvidia-cudnn, cuda-bindings, triton, etc.) that are completely wasted
# on a machine with no GPU. Note: your actual local venv has the full
# CUDA build installed despite having no GPU — that's fine for local dev,
# but not worth replicating in a container image.
RUN pip install --no-cache-dir --timeout=120 --retries 10 torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout=120 --retries 10 -r requirements-docker.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Accept the Coqui TTS license non-interactively at build time — without
# this, XTTS v2's first download attempt blocks waiting for a y/n prompt
# that will never come inside a container build.
ENV COQUI_TOS_AGREED=1
# Keeps Hugging Face / Coqui model downloads from printing progress bars
# that spam build logs.
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# 🚀 THE KEY STEP: pre-download every model weight at BUILD time, not on
# the first real request. Without this, your first API call after a
# container starts pays for downloading faster-whisper (~500MB), XTTS v2
# (~1.8GB), and the sentence-transformers embedding model (~90MB) all at
# once — easily several minutes, which is exactly the kind of delay that
# trips a host's request timeout and looks like a broken deployment.
#
# 🚀 THE FIX (restructured): calls the underlying libraries directly
# (faster_whisper.WhisperModel, TTS.api.TTS, HuggingFaceEmbeddings) with
# the exact same model names/params your actual services use — instead
# of importing src/stt_service.py etc. This means this layer depends only
# on requirements-docker.txt, NOT on your application code. Placed BEFORE
# `COPY . .` below, so editing api.py, App.jsx-equivalent backend files,
# etc. never invalidates this layer again — only a real dependency change
# forces the ~2GB re-download from here on.
RUN python -c "\
print('[build] Warming up STT (faster-whisper)...'); \
from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8'); \
print('[build] Warming up TTS (XTTS v2)...'); \
from TTS.api import TTS; TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False); \
print('[build] Warming up embeddings (sentence-transformers)...'); \
from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'); \
print('[build] All models cached.') \
"

# Now copy the actual application code. Only this layer (and CMD) needs
# to rerun when you edit src/*.py — the expensive stuff above stays cached.
COPY . .

# Runtime-only data should NOT be baked into the image — mount these as
# volumes instead (see the docker run command). Declaring them here is
# documentation, not enforcement, but makes the intent explicit.
VOLUME ["/app/outputs", "/app/logs", "/app/data/reference_voices"]

EXPOSE 8000

# --host 0.0.0.0 is required inside a container — 127.0.0.1 (the default)
# would only accept connections from inside the container itself, making
# it unreachable even with the port exposed/published.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
