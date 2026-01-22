"""FastAPI service that proxies Gemini for language detection."""

from __future__ import annotations

import logging
import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
)
log = logging.getLogger('llm_service')

app = FastAPI(title='LLM Language Detection Service', version='0.1.0')


class DetectRequest(BaseModel):
    text: str = Field(..., description='Review text to classify.')


class DetectResponse(BaseModel):
    language: str
    # raw_response: Any | None = None


session = requests.Session()

SYSTEM_PROMPT = (
    'You are a language detector specialized in user-written gym reviews.\n'
    'Most texts come from gyms in Spain, so Spanish (es) is the most common language, '
    'but any language is possible.\n'
)

USER_PROMPT_TEMPLATE = (
    '\n'
    'Task:\n'
    '- Detect the primary natural language of the review text.\n'
    '- Return ONLY a two-letter ISO 639-1 language code (e.g., es, en, pt, fr).\n'
    '- Do NOT return explanations, probabilities, or any extra text.\n'
    '\n'
    'Examples:\n'
    'Text: ```de 10```\n'
    'Language: es\n'
    '\n'
    'Text: ```top gym, muito bom```\n'
    'Language: pt\n'
    '\n'
    'Now classify this text:\n'
    'Text: ```{text}```\n'
    'Language code:'
)


def _call_gemini(text: str) -> str:
    """Invoke Gemini API to detect language."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise RuntimeError('GOOGLE_API_KEY is not set.')

    model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    endpoint = (
        f'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{model}:generateContent?key={api_key}'
    )

    payload = {
        'systemInstruction': {'parts': [{'text': SYSTEM_PROMPT}]},
        'contents': [{'parts': [{'text': USER_PROMPT_TEMPLATE.format(text=text)}]}],
        'generationConfig': {'temperature': 0.0},
    }

    response = session.post(endpoint, json=payload, timeout=30)
    log_fn = log.info if response.status_code < 400 else log.error
    log_fn('Gemini status %s response', response.status_code)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f'Gemini error: {response.status_code}',
        )

    try:
        data = response.json()
    except ValueError as exc:
        log.error('Gemini JSON decode failed: %s', exc)
        raise HTTPException(status_code=502, detail='Invalid Gemini JSON.') from exc
    try:
        text_response = data['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError) as exc:
        log.error('Unexpected Gemini response structure: %s', data)
        raise HTTPException(status_code=502, detail='Unexpected Gemini response.') from exc

    return text_response.strip().lower()


def _call_mistral(text: str) -> str:
    """Invoke Mistral chat-completions API to detect language."""
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        raise RuntimeError('MISTRAL_API_KEY is not set.')

    model = os.getenv('MISTRAL_MODEL', 'mistral-large-2411')
    endpoint = 'https://api.mistral.ai/v1/chat/completions'
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(text=text)},
    ]
    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0.0,
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    response = session.post(endpoint, json=payload, headers=headers, timeout=30)
    log_fn = log.info if response.status_code < 400 else log.error
    log_fn('Mistral status %s response', response.status_code)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=f'Mistral error: {response.status_code}',
        )

    try:
        data = response.json()
    except ValueError as exc:
        log.error('Mistral JSON decode failed: %s', exc)
        raise HTTPException(status_code=502, detail='Invalid Mistral JSON.') from exc

    try:
        content = data['choices'][0]['message']['content']
    except (KeyError, IndexError) as exc:
        log.error('Unexpected Mistral response structure: %s', data)
        raise HTTPException(status_code=502, detail='Unexpected Mistral response.') from exc

    return content.strip().lower()


@app.post('/detect_language', response_model=DetectResponse)
def detect_language(request: DetectRequest) -> DetectResponse:
    """HTTP endpoint that returns the language code for the provided text."""
    language_code = _call_mistral(request.text)
    return DetectResponse(language=language_code)
