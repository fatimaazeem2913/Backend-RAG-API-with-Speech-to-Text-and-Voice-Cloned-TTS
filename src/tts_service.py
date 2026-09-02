import os
import re
import time
import logging
import numpy as np
import torch
from typing import Generator
from TTS.api import TTS

logger = logging.getLogger("voice_rag")

class TTSService:
    def __init__(self, speaker_wav: str = "data/reference_voices/my_voice.wav"):
        self.speaker_wav = speaker_wav
        logger.info("[TTS Service] Initializing XTTS v2...")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

        # 🚀 XTTS on CPU is very slow (5-50s per sentence is normal). If a
        # CUDA GPU is available, use it — typically a 10-20x speedup.
        if torch.cuda.is_available():
            self.tts.to("cuda")
            logger.info(f"[TTS Service] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("[TTS Service] No CUDA GPU detected — running on CPU. Per-sentence synthesis will be slow; this is a hardware limit, not a code issue.")

        # 🚀 THE FIX: precompute the speaker's voice embedding ONCE at startup
        # instead of recomputing it from the reference wav on every single
        # sentence call. Recomputing conditioning latents is expensive and
        # your voice doesn't change between sentences — this was very likely
        # the single biggest source of latency in the old code.
        self._model = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self._supports_streaming = False

        try:
            self._model = self.tts.synthesizer.tts_model
            if os.path.exists(self.speaker_wav):
                t0 = time.time()
                self.gpt_cond_latent, self.speaker_embedding = self._model.get_conditioning_latents(
                    audio_path=[self.speaker_wav]
                )
                logger.info(f"[TTS Service] Speaker latents precomputed in {time.time() - t0:.2f}s")
                self._supports_streaming = hasattr(self._model, "inference_stream")
                if self._supports_streaming:
                    logger.info("[TTS Service] Chunk-level streaming inference available.")
                else:
                    logger.info("[TTS Service] inference_stream not found on this TTS version — using cached-latent whole-sentence inference instead.")
            else:
                logger.warning(f"[TTS Service] Reference voice not found at {self.speaker_wav} — falling back to default per-call synthesis.")
        except Exception as e:
            logger.warning(f"[TTS Service] Could not precompute speaker latents, falling back to per-call synthesis: {e}")
            self._model = None

        logger.info("[TTS Service] ✅ XTTS v2 ready.")

    def _clean_for_speech(self, text: str) -> str:
        if not text:
            return ""

        # 1. Remove citations
        cleaned = re.sub(r'\[Source:[^\]]+\]', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[[^\]]*Page:[^\]]*\]', '', cleaned, flags=re.IGNORECASE)

        # 2. Convert mathematical symbols to spoken English
        cleaned = cleaned.replace(r'\sqrt', ' square root of ')
        cleaned = cleaned.replace(r'\sum', ' sum of ')
        cleaned = cleaned.replace(r'\times', ' times ')
        cleaned = cleaned.replace(r'\mu', ' centroid ')
        cleaned = cleaned.replace(r'\forall', ' for all ')
        cleaned = cleaned.replace(r'\in', ' in ')
        cleaned = cleaned.replace(r'\cup', ' union ')
        cleaned = cleaned.replace(r'\le', ' less than or equal to ')
        cleaned = cleaned.replace(r'\ge', ' greater than or equal to ')
        cleaned = cleaned.replace('^2', ' squared ')
        cleaned = cleaned.replace('^', ' to the power of ')

        # 3. Clean raw markdown symbols, formulas, and brackets
        cleaned = re.sub(r'---', ' ', cleaned)
        cleaned = re.sub(r'[#*`_~]', ' ', cleaned)
        cleaned = re.sub(r'\$+', '', cleaned)
        cleaned = re.sub(r'[\{\}\(\)\[\]\\]', ' ', cleaned)

        # 4. Clean extra punctuation and spaces
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        cleaned = re.sub(r'\.{2,}', '.', cleaned)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _split_into_chunks(self, text: str, max_len: int = 100) -> list:
        """Splits on sentence boundaries first, then further splits any long
        sentence on clause boundaries (commas/semicolons). Shorter, more
        uniform chunks synthesize faster individually, which shrinks the
        worst-case gap between streamed audio segments."""
        sentences = [s.strip() for s in re.split(r'(?<=[.?!:\n])\s+', text) if len(s.strip()) > 3]
        chunks = []
        for s in sentences:
            if len(s) <= max_len:
                chunks.append(s)
                continue
            parts = re.split(r'(?<=[,;])\s+', s)
            buf = ""
            for p in parts:
                if buf and len(buf) + 1 + len(p) > max_len:
                    chunks.append(buf)
                    buf = p
                else:
                    buf = f"{buf} {p}".strip()
            if buf:
                chunks.append(buf)
        return chunks

    def _tensor_to_pcm16_bytes(self, chunk) -> bytes:
        arr = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.array(chunk)
        pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm.tobytes()

    def stream_audio(self, text: str, session_id: str = "default_session", timeline=None) -> Generator[bytes, None, None]:
        """Synthesizes text sentence-by-sentence so audio playback begins immediately."""
        if not text.strip():
            return

        cleaned_text = self._clean_for_speech(text)
        if not cleaned_text:
            return

        sentences = self._split_into_chunks(cleaned_text)

        if timeline:
            timeline.mark(
                "tts.synthesis_started",
                language="en", input_chars=len(text), spoken_chars=len(cleaned_text), sentences=len(sentences),
            )

        total_bytes = 0
        total_chunks = 0
        first_chunk_marked = False

        def _maybe_mark_first_chunk(data: bytes):
            nonlocal first_chunk_marked
            if not first_chunk_marked and timeline:
                first_chunk_marked = True
                timeline.mark("tts.first_chunk_ready", bytes=len(data), audio_s=round(len(data) / (24000 * 2), 3))

        for idx, sentence in enumerate(sentences):
            try:
                produced_any = False

                # Tier 1: real chunk-level streaming (lowest time-to-first-audio)
                if self._supports_streaming and self.gpt_cond_latent is not None:
                    try:
                        for chunk in self._model.inference_stream(
                            sentence, "en", self.gpt_cond_latent, self.speaker_embedding,
                            enable_text_splitting=False,
                        ):
                            data = self._tensor_to_pcm16_bytes(chunk)
                            total_bytes += len(data)
                            total_chunks += 1
                            produced_any = True
                            _maybe_mark_first_chunk(data)
                            yield data
                    except Exception as e:
                        logger.warning(f"[TTS] session={session_id} inference_stream failed on sentence {idx+1}, disabling streaming for remainder of this response: {e}")
                        produced_any = False
                        # Don't keep retrying a call that's incompatible with
                        # this environment's TTS/transformers version — skip
                        # it for the rest of this stream (and future ones).
                        self._supports_streaming = False

                # Tier 2: cached-latent whole-sentence inference (fast, no re-embedding)
                if not produced_any and self._model is not None and self.gpt_cond_latent is not None:
                    out = self._model.inference(sentence, "en", self.gpt_cond_latent, self.speaker_embedding)
                    data = self._tensor_to_pcm16_bytes(out["wav"])
                    total_bytes += len(data)
                    total_chunks += 1
                    produced_any = True
                    _maybe_mark_first_chunk(data)
                    yield data

                # Tier 3: original safe fallback (re-embeds every call, slowest, always works)
                if not produced_any:
                    wav = self.tts.tts(
                        text=sentence, language="en",
                        speaker_wav=self.speaker_wav if os.path.exists(self.speaker_wav) else None
                    )
                    pcm_array = (np.clip(np.array(wav), -1.0, 1.0) * 32767).astype(np.int16)
                    data = pcm_array.tobytes()
                    total_bytes += len(data)
                    total_chunks += 1
                    _maybe_mark_first_chunk(data)
                    yield data

                if timeline:
                    timeline.mark("tts.sentence_done", n=idx + 1, of=len(sentences), chunks_so_far=total_chunks)
            except Exception as e:
                logger.error(f"[TTS] session={session_id} synthesis error on sentence '{sentence[:30]}...': {e}", exc_info=True)
                continue

        if timeline:
            # 16-bit mono PCM at 24kHz -> 2 bytes/sample -> bytes / (24000*2) = seconds of audio
            audio_s = total_bytes / (24000 * 2)
            synthesis_s = timeline.elapsed()
            realtime_factor = round(synthesis_s / audio_s, 3) if audio_s > 0 else 0.0
            timeline.mark(
                "tts.last_chunk_ready",
                chunks=total_chunks, audio_s=round(audio_s, 3), synthesis_s=round(synthesis_s, 3),
                realtime_factor=realtime_factor,
            )