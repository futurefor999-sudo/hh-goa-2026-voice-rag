"""
Tests SarvamSTT's request construction and error handling by mocking
requests.post — this sandbox has no network access, so the actual
Sarvam API can't be called here. What's verified instead: given a
canned HTTP response shaped like Sarvam's documented API (or a failure
mode of it), does our client build the right request and handle the
response correctly? That's our code; the network call itself isn't.

To exercise the real network path once you have a key:
    SARVAM_API_KEY=... python -m scripts.run_query --index data/index.pkl --audio clip.wav
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stt.sarvam_stt import SarvamSTT, SarvamSTTError, MockSTT, get_stt_client, _guess_content_type


def test_guess_content_type_known_extensions():
    assert _guess_content_type("clip.wav") == "audio/wav"
    assert _guess_content_type("clip.webm") == "audio/webm"
    assert _guess_content_type("clip.mp3") == "audio/mpeg"


def test_guess_content_type_unknown_extension_falls_back():
    assert _guess_content_type("clip.xyz") == "application/octet-stream"
    assert _guess_content_type("no_extension") == "application/octet-stream"


def test_sarvam_stt_requires_api_key():
    try:
        SarvamSTT(api_key="")
        assert False, "should have raised"
    except SarvamSTTError:
        pass


@patch("src.stt.sarvam_stt.requests.post")
def test_sarvam_stt_success_parses_transcript(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"transcript": "  what is the repo rate  "}
    mock_post.return_value = mock_resp

    client = SarvamSTT(api_key="fake-key", model="saarika:v2")
    result = client.transcribe(b"fake-audio-bytes", filename="q.webm")

    assert result == "what is the repo rate"
    # verify the request was actually built correctly
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["api-subscription-key"] == "fake-key"
    assert kwargs["data"]["model"] == "saarika:v2"
    assert kwargs["files"]["file"][0] == "q.webm"
    assert kwargs["files"]["file"][2] == "audio/webm"


@patch("src.stt.sarvam_stt.requests.post")
def test_sarvam_stt_non_200_raises(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "invalid api key"
    mock_post.return_value = mock_resp

    client = SarvamSTT(api_key="fake-key")
    try:
        client.transcribe(b"audio", filename="q.wav")
        assert False, "should have raised"
    except SarvamSTTError as e:
        assert "401" in str(e)


@patch("src.stt.sarvam_stt.requests.post")
def test_sarvam_stt_missing_transcript_field_raises(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"some_other_field": "oops"}
    mock_post.return_value = mock_resp

    client = SarvamSTT(api_key="fake-key")
    try:
        client.transcribe(b"audio", filename="q.wav")
        assert False, "should have raised"
    except SarvamSTTError as e:
        assert "transcript" in str(e)


@patch("src.stt.sarvam_stt.requests.post")
def test_sarvam_stt_network_error_wrapped(mock_post):
    import requests
    mock_post.side_effect = requests.ConnectionError("no route to host")

    client = SarvamSTT(api_key="fake-key")
    try:
        client.transcribe(b"audio", filename="q.wav")
        assert False, "should have raised"
    except SarvamSTTError as e:
        assert "request failed" in str(e).lower()


def test_mock_stt_decodes_bytes_as_text():
    stt = MockSTT()
    result = stt.transcribe(b"hello world", filename="ignored.wav")
    assert result == "hello world"


def test_get_stt_client_mock_provider():
    client = get_stt_client("mock", api_key=None, model="unused")
    assert isinstance(client, MockSTT)


def test_get_stt_client_sarvam_without_key_raises():
    try:
        get_stt_client("sarvam", api_key=None, model="saarika:v2")
        assert False, "should have raised"
    except SarvamSTTError:
        pass


def test_get_stt_client_unknown_provider_raises():
    try:
        get_stt_client("elevenlabs", api_key="x", model="y")
        assert False, "should have raised"
    except ValueError:
        pass


if __name__ == "__main__":
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
