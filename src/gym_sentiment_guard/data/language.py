"""Language identification utilities."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import os
from pathlib import Path

import fasttext
import pandas as pd
import requests

from ..utils import get_logger, json_log

log = get_logger(__name__)

_SPANISH_LABEL = 'es'
_PREDICT_TOP_K = 3
_FALLBACK_ENABLED = False
_FALLBACK_THRESHOLD = 0.75
_FALLBACK_ENDPOINT: str | None = None
_FALLBACK_API_KEY_ENV: str | None = None


def configure_language_fallback(
    *,
    enabled: bool,
    threshold: float,
    endpoint: str | None,
    api_key_env: str | None,
) -> None:
    """Configure the LLM fallback settings without changing the public signature."""
    global _FALLBACK_ENABLED, _FALLBACK_THRESHOLD, _FALLBACK_ENDPOINT, _FALLBACK_API_KEY_ENV
    _FALLBACK_ENABLED = enabled
    _FALLBACK_THRESHOLD = threshold
    _FALLBACK_ENDPOINT = endpoint
    _FALLBACK_API_KEY_ENV = api_key_env


class LanguageFilterError(RuntimeError):
    """Raised when language filtering cannot be completed."""


@lru_cache(maxsize=1)
def _load_fasttext_model(model_path: str) -> fasttext.FastText._FastText:
    """Load and cache the fastText language identification model."""
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f'fastText model not found at: {path_obj}')
    return fasttext.load_model(str(path_obj))


def _normalize_label(raw_label: Sequence[str] | Sequence[Sequence[str]]) -> str:
    """Extract the language code from fastText labels."""
    if not raw_label:
        return ''
    label = raw_label[0] if isinstance(raw_label[0], str) else raw_label[0][0]
    return label.replace('__label__', '') if label else ''


def _predict_languages(
    texts: Sequence[str],
    model_path: Path,
    batch_size: int = 512,
) -> tuple[list[str], list[float], list[float]]:
    """Detect languages and their confidences using the cached fastText model."""
    model = _load_fasttext_model(str(model_path))
    predictions = [''] * len(texts)
    predicted_confidences = [0.0] * len(texts)
    spanish_probs = [0.0] * len(texts)

    batch: list[str] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        if not batch:
            return
        labels_list, probs_list = model.predict(batch, k=_PREDICT_TOP_K)
        for idx, labels, probs in zip(batch_indices, labels_list, probs_list, strict=True):
            predictions[idx] = _normalize_label(labels)
            predicted_confidences[idx] = (
                float(probs[0]) if len(probs) > 0 else 0.0
            )
            spanish_prob = 0.0
            for label, probability in zip(labels, probs, strict=True):
                normalized = label.replace('__label__', '')
                if normalized == _SPANISH_LABEL:
                    spanish_prob = float(probability)
                    break
            spanish_probs[idx] = spanish_prob
        batch.clear()
        batch_indices.clear()

    for idx, text in enumerate(texts):
        normalized = text.replace('\r', ' ').replace('\n', ' ').strip()
        if not normalized:
            continue
        normalized = ' '.join(normalized.split())
        batch.append(normalized)
        batch_indices.append(idx)
        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()
    return predictions, predicted_confidences, spanish_probs


def filter_spanish_comments(
    input_csv: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    text_column: str = 'comment',
    batch_size: int = 512,
    output_path: str | Path | None = None,
    rejected_output_path: str | Path | None = None,
) -> Path:
    """
    Filter a CSV to rows whose comment column is detected as Spanish.

    Args:
        input_csv: Path to the raw CSV file.
        output_dir: Directory where the filtered CSV will be written.
        model_path: Path to the fastText `lid.176.bin` model.
        text_column: Name of the column containing textual reviews.
        batch_size: Number of records per fastText inference batch.
        output_path: Optional explicit path for the Spanish-only CSV.
        rejected_output_path: Optional explicit path for the non-Spanish CSV.

    Returns:
        Path to the filtered CSV written under ``output_dir`` with the same file name.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_path}')

    output_path = Path(output_dir) / input_path.name if output_path is None else Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rejected_output_path is None:
        rejected_output_path = output_path.with_name(
            f'{output_path.stem}.non_spanish{output_path.suffix}',
        )
    else:
        rejected_output_path = Path(rejected_output_path)
    rejected_output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise LanguageFilterError(
            f"Column '{text_column}' not found in CSV {input_path.name}",
        )

    original_rows = len(df)
    comments = df[text_column].fillna('').astype(str).tolist()
    languages, confidences, spanish_probs = _predict_languages(
        comments,
        Path(model_path),
        batch_size=batch_size,
    )
    fallback_requests = 0
    if _FALLBACK_ENABLED and _FALLBACK_ENDPOINT:
        for idx, conf in enumerate(confidences):
            if conf >= _FALLBACK_THRESHOLD:
                continue
            text = comments[idx]
            if not text.strip():
                continue
            llm_language = _call_llm_language_detector(text)
            if llm_language:
                languages[idx] = llm_language
                fallback_requests += 1

    spanish_mask = pd.Series(
        [lang == 'es' for lang in languages],
        index=df.index,
    )
    probability_series = pd.Series(spanish_probs, index=df.index, name='es_confidence')
    filtered = df.loc[spanish_mask].copy()
    rejected = df.loc[~spanish_mask].copy()
    rejected['es_confidence'] = probability_series.loc[rejected.index].values

    filtered.to_csv(output_path, index=False)
    rejected.to_csv(rejected_output_path, index=False)

    log.info(
        json_log(
            'language_filter.completed',
            component='data.language',
            input=str(input_path),
            output=str(output_path),
            rows_in=original_rows,
            rows_out=len(filtered),
            rows_rejected=len(rejected),
            kept_language='es',
            rejected_output=str(rejected_output_path),
            fallback_requests=fallback_requests,
        ),
    )

    return output_path


def _call_llm_language_detector(text: str) -> str | None:
    """Call the configured LLM endpoint to classify a review's language."""
    if not _FALLBACK_ENDPOINT:
        return None

    headers = {'Content-Type': 'application/json'}
    if _FALLBACK_API_KEY_ENV:
        api_key = os.getenv(_FALLBACK_API_KEY_ENV)
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

    payload = {'text': text}
    try:
        response = requests.post(_FALLBACK_ENDPOINT, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        log.warning(
            json_log(
                'language_filter.llm_error',
                component='data.language',
                error=str(exc),
            ),
        )
        return None

    log.debug(
        json_log(
            'language_filter.llm_response',
            component='data.language',
            status=response.status_code,
            response=response.text[:500],
        ),
    )

    if response.status_code >= 400:
        return None

    try:
        data = response.json()
    except ValueError:
        return None

    if isinstance(data, dict):
        language = (
            data.get('language')
            or data.get('language_code')
            or data.get('languageTag')
            or data.get('result')
        )
        if isinstance(language, str):
            return language.strip().lower()
    return None
