"""
Sarvam speech-to-text client.

Wraps Sarvam's /speech-to-text REST endpoint. Kept intentionally thin
(single responsibility: audio bytes in, transcript text out) so the
harness in src/pipeline.py owns retries/error-handling/timing uniformly
across every stage rather than each client rolling its own.

Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
(check there for the current endpoint path/model names — Sarvam's API
has moved before; SARVAM_STT_MODEL is configurable in .env precisely
so a naming change doesn't require touching code.)
"""
from __future__ import annotations
import requests

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTTError(Exception):
    pass


class SarvamSTT:
    def __init__(self, api_key: str, model: str = "saarika:v2", timeout: float = 15.0):
        if not api_key:
            raise SarvamSTTError("SARVAM_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav", language_code: str = "unknown") -> str:
        """Send raw audio bytes to Sarvam, return the transcript string.
        Raises SarvamSTTError on any non-200 response or malformed body —
        the caller (pipeline harness) decides whether to retry.

        Sarvam's REST /speech-to-text endpoint auto-detects the codec
        from most common containers (WAV, MP3, OGG, WebM, M4A, FLAC,
        AAC, ...), so the browser's native MediaRecorder output
        (audio/webm) can be sent as-is — no client-side transcoding
        needed. Content-type is guessed from the filename extension
        purely for the multipart part's declared type; Sarvam still
        inspects the actual bytes."""
        headers = {"api-subscription-key": self.api_key}
        data = {"model": self.model, "language_code": language_code}
        content_type = _guess_content_type(filename)
        files = {"file": (filename, audio_bytes, content_type)}

        try:
            resp = requests.post(
                SARVAM_STT_URL, headers=headers, data=data, files=files, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise SarvamSTTError(f"Sarvam STT request failed: {e}") from e

        if resp.status_code != 200:
            raise SarvamSTTError(f"Sarvam STT returned {resp.status_code}: {resp.text[:500]}")

        try:
            body = resp.json()
        except ValueError as e:
            raise SarvamSTTError(f"Sarvam STT returned non-JSON body: {resp.text[:500]}") from e

        transcript = body.get("transcript")
        if transcript is None:
            raise SarvamSTTError(f"Sarvam STT response missing 'transcript': {body}")
        return transcript.strip()


_CONTENT_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
    ".webm": "audio/webm", ".m4a": "audio/mp4", ".flac": "audio/flac",
    ".aac": "audio/aac", ".opus": "audio/opus", ".amr": "audio/amr",
}


def _guess_content_type(filename: str) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


class MockSTT:
    """Used in tests / local dev when no Sarvam key is configured, and
    for the CLI text-query path where audio isn't involved at all."""

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav", language_code: str = "unknown") -> str:
        # In mock mode we expect the "audio_bytes" to actually be the
        # UTF-8 encoded query text, so scripts/run_query.py and tests
        # can exercise the full pipeline without a real audio file.
        return audio_bytes.decode("utf-8").strip()


def get_stt_client(provider: str, api_key: str | None, model: str):
    if provider == "sarvam":
        if not api_key:
            raise SarvamSTTError(
                "STT_PROVIDER=sarvam but SARVAM_API_KEY is not set. "
                "Set it in .env, or set STT_PROVIDER=mock for local testing."
            )
        return SarvamSTT(api_key=api_key, model=model)
    if provider == "mock":
        return MockSTT()
    raise ValueError(f"unknown STT provider: {provider} (supported: sarvam, mock)")
