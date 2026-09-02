import os
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

_configured = False


def configure(log_dir: str):
    """Call once at app startup with the same log directory the rest of the
    app already uses. Safe to call more than once — only configures handlers
    the first time."""
    global _configured
    if _configured:
        return
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    _configured = True


class Timeline:
    """Per-pipeline-stage event tracer.

    Each mark() logs one line:
        [request_id] event.name              t+   Xms (+  Yms) key=value ...

    - t+   is elapsed time since this Timeline was created.
    - (+ ) is the delta since the previous mark on THIS Timeline.

    A new pipeline stage (e.g. the chat stage vs. the TTS stage) should
    create its own Timeline so its t+ clock starts fresh at 0 for that
    stage — but pass the same request_id from an earlier stage's Timeline
    to correlate them together when grepping logs/app.log.
    """

    def __init__(self, stage: str, request_id: str = None):
        self.stage = stage
        self.request_id = request_id or uuid.uuid4().hex[:8]
        self._start = time.time()
        self._last = self._start

    def mark(self, event: str, **kwargs) -> float:
        now = time.time()
        t_plus_ms = (now - self._start) * 1000
        delta_ms = (now - self._last) * 1000
        self._last = now

        kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
        line = f"[{self.request_id}] {event:<28} t+ {t_plus_ms:6.0f}ms (+ {delta_ms:6.0f}ms) {kv}".rstrip()
        logger.info(line)

        return now - self._start

    def elapsed(self) -> float:
        return time.time() - self._start