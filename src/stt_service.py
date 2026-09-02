import os
import tempfile
import numpy as np
from faster_whisper import WhisperModel

class STTService:
    def __init__(self, model_size: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
        print(f"[STT Service] Loading faster-whisper '{model_size}' ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # 🚀 Vocabulary hint: nudges Whisper toward domain terms it otherwise
        # mishears (e.g. "Theil's" -> "Harald's"). Safe to use here because
        # transcribe() below has a dedup guard against the parroting failure
        # mode this caused earlier.
        self.prompt = (
            "Machine learning and statistics terms: K-Means clustering, linear regression, "
            "loss function, MSE, RMSE, MAE, MAPE, centroids, algorithms, Theil's U statistic, "
            "relative measure of accuracy, cross validation, confusion matrix, precision, recall, "
            "F1 score, overfitting, regularization, gradient descent."
        )
        
        # 🚀 Blacklist classic Whisper silence hallucinations
        self.banned_phrases = ["thank you.", "thank you", "i love this.", "i love this", "thanks.", "bye.", "you", "."]
        
        print("[STT Service] ✅ Whisper model ready.")

    def _looks_like_prompt_echo(self, text: str) -> bool:
        """Detects the failure mode where Whisper just parrots the vocabulary
        hint back instead of transcribing real speech."""
        t = text.lower().strip().strip(".")
        p = self.prompt.lower()
        return len(t) >= 12 and t in p

    def transcribe(self, audio_bytes: bytes, suffix: str = ".webm") -> str:
        if not audio_bytes or len(audio_bytes) < 1000:
            return ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            def _run(use_prompt: bool):
                segs, _ = self.model.transcribe(
                    tmp_path,
                    language="en",
                    beam_size=3,
                    temperature=0.0,
                    initial_prompt=self.prompt if use_prompt else None,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                pieces = [seg.text.strip() for seg in segs if seg.text.strip()]
                deduped = []
                for p in pieces:
                    if not deduped or deduped[-1].lower() != p.lower():
                        deduped.append(p)
                return " ".join(deduped).strip()

            text = _run(use_prompt=True)

            # Safety net: prompt caused Whisper to just echo the vocabulary
            # hint instead of real speech -> retry once with no prompt at all.
            if self._looks_like_prompt_echo(text):
                text = _run(use_prompt=False)

            if not text or text.lower() in self.banned_phrases:
                return ""
            return text
        except Exception:
            return ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe_stream(self, audio_data: np.ndarray) -> str:
        # Slightly stricter silence check to ignore mic hiss
        rms = np.sqrt(np.mean(np.square(audio_data)))
        if rms < 0.015:
            return ""

        try:
            segments, _ = self.model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=self.prompt,  # Forces Whisper to listen for ML terms
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300)
            )
            text = " ".join([seg.text.strip() for seg in segments]).strip()

            # Filter out the garbage phrases instantly
            if not text or text.lower() in self.banned_phrases:
                return ""

            return text
        except Exception as e:
            print(f"[STT Stream Error]: {e}")
            return ""