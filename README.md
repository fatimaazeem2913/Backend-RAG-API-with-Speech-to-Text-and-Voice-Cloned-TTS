# Backend — RAG API with Speech-to-Text and Voice-Cloned TTS

FastAPI service that answers questions over an ingested document corpus using
four interchangeable retrieval strategies (Dense, BM25, Hybrid RRF,
Hierarchical), transcribes spoken questions with `faster-whisper`, and
streams the answer back as speech in a cloned voice using Coqui XTTS v2.

Runs as a fully independent process from the frontend — no shared build
step. The client lives in its own repository:
[day-23-voice-cloning-tts-frontend](../../day-23-voice-cloning-tts-frontend)
*(update this link to your actual frontend repo name)*.

## Requirements

- Python 3.10+
- `ffmpeg` installed at the system level — `faster-whisper` and `TTS` both shell out to it for audio decoding
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/))
- ~3–4GB free disk for model weights (`faster-whisper small.en`, XTTS v2, the sentence-transformers embedding model)
- **Recommended: a CUDA GPU.** XTTS v2 runs 10–20x faster on GPU. On CPU-only machines, expect 5–50 seconds of synthesis *per sentence* — this is a hardware ceiling, not a bug (see [Known Limitations](#known-limitations)). The server auto-detects CUDA at startup and logs which device it's using.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/` (see [Environment Variables](#environment-variables) below).

You'll also need a short (~10–30s) reference clip of your own voice at:

```
backend/data/reference_voices/my_voice.wav
```

This is what XTTS v2 clones from. If the file is missing, the service falls back to XTTS's default voice and logs a warning rather than failing.

## Running

```bash
uvicorn src.api:app --reload --port 8000
```

On startup you should see, in order:

```
[STT Service] ✅ Whisper model ready.
[TTS Service] Speaker latents precomputed in X.XXs
[TTS Service] ✅ XTTS v2 ready.
INFO:     Application startup complete.
```

The document corpus loads automatically on first run from
`data/sample_corpus.json` into a local ChromaDB store at
`outputs/chroma_db/` — no manual ingest step is required. `src/ingestion.py`
(`DocumentIngestionEngine`) contains PDF/DOCX/TXT parsing logic for a future
file-upload flow, but **it is not currently wired to any API endpoint**.

Server runs at `http://127.0.0.1:8000`.

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini API key used for query rewriting and answer generation. `GOOGLE_API_KEY` is also accepted as a fallback name. |
| `COQUI_TOS_AGREED` | Yes | Set to `1`. Coqui TTS refuses to download the XTTS v2 model weights on first run without explicit license acceptance via this variable. |

```env
GEMINI_API_KEY=your_key_here
COQUI_TOS_AGREED=1
```

⚠️ Never commit your real `.env` — make sure it's in `.gitignore` before pushing.

## Endpoints

| Method | Path | Purpose | Used by current frontend? |
|---|---|---|---|
| POST | `/api/transcribe` | Upload an audio blob (webm/ogg) → Whisper transcript | ✅ Yes |
| POST | `/api/rag/chat/stream` | Question in → SSE token-by-token answer, plus citations + standalone query + `request_id` | ✅ Yes |
| POST | `/api/tts/stream` | Streams the answer as chunked raw 24kHz mono PCM16 in the cloned voice | ✅ Yes |
| POST | `/api/rag/chat` | Same retrieval/generation, but returns one final JSON response instead of SSE | ⚠️ Implemented, not currently called |
| POST | `/api/transcribe/stream-chunk` | Incremental chunk transcription for a live-streaming STT flow | ⚠️ Implemented, not currently called |
| WS | `/ws/transcribe` | Live raw PCM16 audio in, incremental partial transcripts out | ⚠️ Implemented, not currently called |

The WebSocket and the non-streaming chat endpoint were both used by the
frontend at earlier points; the app has since moved to SSE streaming for
the chat response (visible token-by-token in the UI as it generates) while
sticking with a simple record-then-transcribe flow for STT rather than the
WebSocket. All of them are left in place as working, tested code rather
than removed.

`/api/tts/stream` returns raw little-endian PCM16 frames, not a playable audio
file — the client is responsible for decoding it (see the frontend README).

## Driving it with curl

```bash
curl -X POST http://127.0.0.1:8000/api/transcribe \
  -F "file=@recording.webm"

# SSE streaming (what the frontend actually calls)
curl -N -X POST http://127.0.0.1:8000/api/rag/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What is k-means clustering?","strategy":"hybrid"}'

# Non-streaming variant — same retrieval/generation, one final JSON response
curl -X POST http://127.0.0.1:8000/api/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What is k-means clustering?","strategy":"hybrid"}'

curl -X POST http://127.0.0.1:8000/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is a test.","session_id":"demo","request_id":"demo01"}' \
  --output speech.bin
```

`speech.bin` is not a playable audio file on its own — it's raw 24kHz mono
PCM16 samples with no WAV header. The frontend decodes it directly via the
Web Audio API as it streams in.

## The `/api/rag/chat` request path

`/api/rag/chat/stream` (SSE, used by the frontend) and `/api/rag/chat`
(single JSON response) share the exact same retrieval and prompt-building
logic via one internal helper, so they can't drift out of sync — the only
difference is whether the answer is streamed token-by-token or returned all
at once:

```
message → contextualize (rewrites follow-ups using pronoun resolution,
           only when history exists AND the message contains a pronoun
           like "it"/"they"/"this")
        │
        ▼
retrieve (dense | bm25 | hybrid RRF | hierarchical — selected by the
          "strategy" field, top_k=4)
        │
        ▼
build prompt (context chunks + [Source: file, Page: N] citation labels)
        │
        ▼
generate answer (Gemini, via generate_content_stream — tried across a
                  fallback list of model names until one succeeds)
        │
        ▼
/stream: SSE events → {type: metadata}, {type: token}×N, {type: done}
/chat:   one JSON response → {answer, citations[], standalone_query, request_id}
```

## Pipeline Logging

Every chat and TTS request is traced through `src/pipeline_log.py`'s
`Timeline` class, writing to both the terminal and `backend/logs/app.log`
(rotates at 5MB, keeps 3 backups):

```
14:22:07.104 | INFO  | [a3f9c210] chat.request_received        t+      0ms (+     0ms) method=hybrid message_chars=34
14:22:07.361 | INFO  | [a3f9c210] llm.request_sent             t+      0ms (+     0ms) model=gemini-3.5-flash prompt_chars=6210
14:22:07.938 | INFO  | [a3f9c210] llm.first_chunk_received     t+    577ms (+   577ms) chars=48
14:22:09.815 | INFO  | [a3f9c210] llm.last_chunk_received      t+   2454ms (+  1877ms) chunks=31 answer_chars=812
14:22:09.816 | INFO  | [a3f9c210] chat.answer_ready            t+   2712ms (+     1ms) chunks_retrieved=4 answer_chars=812 llm_error=False
14:22:09.817 | INFO  | [a3f9c210] chat.response_sent           t+   2713ms (+     1ms) total_s=2.713
14:22:10.402 | INFO  | [a3f9c210] tts_stream.request_received  t+      0ms (+     0ms) text_chars=812 language=en
14:22:10.403 | INFO  | [a3f9c210] tts.synthesis_started        t+      0ms (+     0ms) language=en input_chars=812 spoken_chars=703 sentences=5
14:22:13.118 | INFO  | [a3f9c210] tts.first_chunk_ready        t+   2715ms (+  2715ms) bytes=42240 audio_s=0.88
14:22:13.119 | INFO  | [a3f9c210] tts_stream.first_frame_sent  t+   2717ms (+     1ms) bytes=42244
14:22:15.640 | INFO  | [a3f9c210] tts.sentence_done            t+   5237ms (+  2522ms) n=1 of=5 chunks_so_far=4
...
14:22:31.882 | INFO  | [a3f9c210] tts.last_chunk_ready         t+  21479ms (+  1204ms) chunks=23 audio_s=11.24 synthesis_s=21.479 realtime_factor=1.911
14:22:31.884 | INFO  | [a3f9c210] tts_stream.last_frame_sent   t+  21481ms (+     2ms) frames=23 wire_bytes=539412 total_s=21.481
```

### Reading it

- **`llm.last_chunk_received`** — the answer is complete and speech
  synthesis can begin. This is the LLM→TTS handoff. `first_chunk_s` (visible
  as the delta on `llm.first_chunk_received`) separates time-to-first-token
  from total generation time.
- **`tts.first_chunk_ready`** — the earliest instant audio could start
  playing. The gap between this and `tts.synthesis_started` is the silence a
  user sits through after the text appears.
- **`tts.last_chunk_ready`** — synthesis genuinely finished.
- **`realtime_factor`** — synthesis seconds per second of generated audio.
  Below 1.0 means generation outruns playback and audio is gapless;
  **above 1.0 means it cannot keep up** — the direct, measurable cause of
  the choppy CPU playback documented in [Known Limitations](#known-limitations).
  On CPU this typically sits around 1.5–3x.

### Correlating chat and TTS

The chat response includes a `request_id` (an 8-character hex ID minted by
the `chat` request's `Timeline`). The frontend stores it on the message and
sends it back as a `request_id` field in the JSON body of the following
`/api/tts/stream` call — the backend reuses it instead of minting a new one,
so `grep "\[a3f9c210\]" backend/logs/app.log` gives the complete picture of
one turn, both the LLM stage and the TTS stage. Requests that skip the chat
step (a `curl` call straight to `/api/tts/stream` with no `request_id`) get
an auto-generated ID instead, so lines stay correlated there too.

### Adding a stage

```python
from src.pipeline_log import Timeline

timeline = Timeline("some_stage")
timeline.mark("started", filename=name)
# ... do work ...
timeline.mark("finished", count=len(items), total_s=timeline.elapsed())
```

`mark()` returns seconds elapsed since the Timeline started, so a caller
that also needs the number doesn't have to read the clock twice. To
correlate with an existing chat/TTS turn, pass its `request_id` in:
`Timeline("some_stage", request_id=existing_id)`.

## Layout

```
backend/
├── data/
│   ├── reference_voices/my_voice.wav   # your cloned voice source
│   ├── uploads/                        # present, not currently wired to an endpoint
│   ├── sample_corpus.json              # auto-loaded corpus on first run
│   └── evaluation_set.json             # used by main.py's benchmark
├── outputs/                            # ChromaDB store + evaluation results
├── logs/                               # app.log (created at runtime)
├── tests/
│   └── test_api.py                     # see Testing note below
├── src/
│   ├── api.py                          # FastAPI app — all endpoints
│   ├── pipeline_log.py                 # Timeline event tracer
│   ├── rag_service.py                  # contextualize + retrieve + generate
│   ├── strategies.py                   # dense / BM25 / hybrid RRF / hierarchical retrieval
│   ├── stt_service.py                  # faster-whisper transcription
│   ├── tts_service.py                  # XTTS v2 streaming synthesis
│   └── ingestion.py                    # PDF/DOCX/TXT parsing (not yet wired to an endpoint)
├── main.py                             # retrieval strategy benchmark (see Evaluation)
├── test_llm.py                         # standalone script, contents not covered by this README
├── requirements.txt
└── .env
```

## Evaluation

```bash
python main.py
```

Runs every question in `data/evaluation_set.json` against all four retrieval
strategies (`dense`, `bm25`, `hybrid`, `hierarchical`), measuring answer
correctness, citation precision, hallucination rate, and latency. Full
results (per-question and summarized) are written to
`outputs/evaluation_matrix.json`.

## Testing

`tests/test_api.py` exists in the repo but currently targets a few endpoints
that aren't implemented yet in `src/api.py` — `GET /health`,
`GET /api/rag/sources`, `POST /api/rag/session/reset`, and validation that
rejects an empty/whitespace message. Running it as-is will fail those cases.
The multi-turn chat test (`test_multi_turn_chat_and_citations`) exercises
real, working behavior against `/api/rag/chat`. Treat the rest as a spec for
future endpoints rather than a passing suite for now.

## Known Limitations

- **XTTS `inference_stream` is incompatible with some `transformers`
  versions** in this environment (`GPT2InferenceModel` missing
  `_extract_generation_mode_kwargs`). The service detects this automatically
  on the first failed sentence, logs a warning once, and falls back to
  cached-latent whole-sentence inference for the rest of that session —
  audio still generates correctly, just without true sub-sentence streaming.
- **CPU-only synthesis is slow.** XTTS v2 is a large autoregressive model;
  without a GPU, per-sentence latency of 5–50 seconds is normal. See
  `realtime_factor` in the pipeline logs for the measured impact.
- **Total TTS latency scales linearly with answer length.** A long LLM
  answer (e.g. 28 sentences) means the per-sentence CPU cost above gets paid
  28 times — one observed real run measured 451s (7.5 minutes) of synthesis
  for a single 2,514-character answer. Short answers stay in the
  seconds-to-tens-of-seconds range; verbose ones do not.
- **LLM time-to-first-token can vary significantly.** One observed real run
  measured 16.6s between `llm.request_sent` and `llm.first_chunk_received`
  before the rest of the answer streamed normally — worth watching via the
  pipeline logs if it happens consistently rather than as a one-off.
- **Whisper accuracy on uncommon/domain-specific terms** is limited by the
  `small.en` model's training vocabulary. A vocabulary hint
  (`initial_prompt`) biases toward expected terminology, with a
  repetition/echo guard that detects and retries if the hint itself gets
  echoed back instead of real speech.
- `requirements.txt` pins `transformers<4.48.0` — required for compatibility
  with the installed `TTS` package version; a newer `transformers` breaks
  XTTS's internal generation code entirely rather than just disabling
  streaming.

## Author

Fatima Azeem 


