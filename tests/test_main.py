"""
Tests for Document Q&A API
===========================
Run with: pytest tests/

These tests cover:
- Input sanitization (pure function, no external dependencies)
- API endpoint contracts (correct status codes, correct error messages)
- Edge cases (empty input, wrong file type, missing document)

Note: LLM responses are not tested here — that would require mocking
OpenAI and is outside the scope of this demo. The goal is to verify
the API layer behaves correctly before any AI calls are made.
"""

import pytest
from fastapi.testclient import TestClient
from main import app, sanitize_text
import io

# TestClient wraps the FastAPI app and lets us call endpoints
# without starting a real server. Standard FastAPI testing pattern.
client = TestClient(app)


# ── Unit tests: sanitize_text ─────────────────────────────────────────────
# These test the function directly, no HTTP involved.

def test_sanitize_text_removes_control_characters():
    """Invisible control characters should be stripped."""
    dirty = "Hello\x00World\x1f"
    result = sanitize_text(dirty)
    assert result == "HelloWorld"


def test_sanitize_text_keeps_normal_text():
    """Regular text should pass through unchanged."""
    text = "This is a normal sentence."
    assert sanitize_text(text) == text


def test_sanitize_text_keeps_newlines_and_tabs():
    """Newlines and tabs are valid — they should not be removed."""
    text = "Line one\nLine two\tTabbed"
    assert sanitize_text(text) == text


def test_sanitize_text_empty_string():
    """Empty string in, empty string out."""
    assert sanitize_text("") == ""


def test_sanitize_text_strips_whitespace():
    """Leading/trailing whitespace should be stripped."""
    assert sanitize_text("  hello  ") == "hello"


# ── API tests: health check ───────────────────────────────────────────────

def test_root_returns_200():
    """Health check endpoint should always return 200."""
    response = client.get("/")
    assert response.status_code == 200


def test_root_returns_status_field():
    """Health check response should include a 'status' field."""
    response = client.get("/")
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"


# ── API tests: /upload ────────────────────────────────────────────────────

def test_upload_rejects_non_pdf():
    """Uploading a .txt file should return 400."""
    fake_file = io.BytesIO(b"this is not a pdf")
    response = client.post(
        "/upload",
        files={"file": ("document.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file():
    """Uploading an empty PDF should return 400."""
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", empty_file, "application/pdf")}
    )
    assert response.status_code == 400


# ── API tests: /ask ───────────────────────────────────────────────────────

def test_ask_rejects_empty_question():
    """An empty question should return 400, not crash."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_ask_rejects_whitespace_only_question():
    """A question that's only spaces should also return 400."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


# ── API tests: /summarize ─────────────────────────────────────────────────

def test_summarize_rejects_empty_text():
    """Empty text should return 400."""
    response = client.post("/summarize", json={"text": ""})
    assert response.status_code == 400


def test_summarize_rejects_short_text():
    """Text under 50 characters is too short to summarize."""
    response = client.post("/summarize", json={"text": "too short"})
    assert response.status_code == 400


def test_summarize_rejects_whitespace_only():
    """Whitespace-only text should return 400."""
    response = client.post("/summarize", json={"text": "   "})
    assert response.status_code == 400


# ── API tests: /extract-tasks ─────────────────────────────────────────────

def test_extract_tasks_rejects_empty_text():
    """Empty text should return 400."""
    response = client.post("/extract-tasks", json={"text": ""})
    assert response.status_code == 400


def test_extract_tasks_rejects_short_text():
    """Text under 50 characters should return 400."""
    response = client.post("/extract-tasks", json={"text": "do something"})
    assert response.status_code == 400