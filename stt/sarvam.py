"""
stt/sarvam.py
─────────────
Sarvam AI Speech-to-Text wrapper using the saaras:v3 model.

Supports:
- File-based transcription (bytes or file path)
- Microphone capture + transcription via sounddevice

The STT step is intentionally excluded from the 200ms retrieval target —
it requires a network round-trip to Sarvam's API. The benchmark harness
measures it separately.
"""
from __future__ import annotations

import io
import os
import time
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


class SarvamSTT:
    """
    Wrapper around the Sarvam AI STT REST API (saaras:v3).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY not set. Get your key at https://dashboard.sarvam.ai"
            )
        # lazy import — only needed when STT is called
        from sarvamai import SarvamAI
        self._client = SarvamAI(api_subscription_key=self.api_key)
        logger.info("SarvamSTT initialized with saaras:v3")

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mode: str = "transcribe",
        language_code: Optional[str] = None,
    ) -> str:
        """
        Transcribe raw audio bytes.

        Args:
            audio_bytes: Raw audio data (WAV preferred, 16kHz)
            filename:    Hint for MIME type detection
            mode:        'transcribe' | 'translate' | 'verbatim'
            language_code: ISO code hint e.g. 'hi-IN', 'en-IN'. None = auto-detect.

        Returns:
            Transcribed text string.
        """
        t0 = time.perf_counter()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename

        kwargs: dict = dict(
            file=audio_file,
            model="saaras:v3",
            mode=mode,
        )
        if language_code:
            kwargs["language_code"] = language_code

        response = self._client.speech_to_text.transcribe(**kwargs)
        transcript = getattr(response, "transcript", "") or ""
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"STT transcribed in {elapsed_ms:.0f}ms: '{transcript[:80]}...'")
        return transcript.strip()

    def transcribe_file(self, filepath: str, **kwargs) -> str:
        """Transcribe a local audio file by path."""
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        return self.transcribe_bytes(audio_bytes, filename=os.path.basename(filepath), **kwargs)

    def record_and_transcribe(
        self,
        duration_seconds: float = 5.0,
        sample_rate: int = 16000,
        **kwargs,
    ) -> str:
        """
        Record audio from the default microphone for `duration_seconds`,
        save as WAV, and transcribe. Requires `sounddevice` and `scipy`.

        Args:
            duration_seconds: How long to record
            sample_rate: Sample rate (Sarvam works best at 16kHz)

        Returns:
            Transcribed text string.
        """
        import sounddevice as sd
        import scipy.io.wavfile as wav
        import numpy as np

        logger.info(f"Recording {duration_seconds}s at {sample_rate}Hz...")
        audio = sd.rec(
            int(duration_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
        )
        sd.wait()
        logger.info("Recording complete, transcribing...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav.write(tmp.name, sample_rate, audio)
            tmp_path = tmp.name

        try:
            return self.transcribe_file(tmp_path, **kwargs)
        finally:
            os.unlink(tmp_path)
