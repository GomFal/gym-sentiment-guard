"""FastAPI service that proxies Gemini for language detection."""

from __future__ import annotations

import logging
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
log = logging.getLogger("llm_service")

app = FastAPI(title="LLM Language Detection Service", version="0.1.0")


class DetectRequest(BaseModel):
    text: str = Field(..., description="Review text to classify.")


class DetectResponse(BaseModel):
    language: str
    #raw_response: Any | None = None


def _call_gemini(text: str) -> str:
    """Invoke Gemini API to detect language."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"{model}:generateContent?key={api_key}"
    )

    prompt = (
        "You are a language detector. Return ONLY the ISO 639-1 language code for the text.\n"
        f"Text: ```{text}```\n"
        "Language code:"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }

    response = requests.post(endpoint, json=payload, timeout=30)
    log_fn = log.info if response.status_code < 400 else log.error
    log_fn("Gemini status %s response", response.status_code)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Gemini error: {response.status_code}",
        )

    try:
        data = response.json()
    except ValueError as exc:
        log.error("Gemini JSON decode failed: %s", exc)
        raise HTTPException(status_code=502, detail="Invalid Gemini JSON.") from exc
    try:
        text_response = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        log.error("Unexpected Gemini response structure: %s", data)
        raise HTTPException(status_code=502, detail="Unexpected Gemini response.") from exc

    return text_response.strip().lower()


@app.post("/detect_language", response_model=DetectResponse)
def detect_language(request: DetectRequest) -> DetectResponse:
    """HTTP endpoint that returns the language code for the provided text."""
    language_code = _call_gemini(request.text)
    return DetectResponse(language=language_code)
