"""Language identification utilities."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import fasttext
import pandas as pd

from ..utils import get_logger, json_log

log = get_logger(__name__)

_SPANISH_LABEL = 'es'
_PREDICT_TOP_K = 3


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
) -> tuple[list[str], list[float]]:
    """Detect languages and Spanish probabilities using the fastText model."""
    model = _load_fasttext_model(str(model_path))
    predictions = [''] * len(texts)
    spanish_probs = [0.0] * len(texts)

    batch: list[str] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        if not batch:
            return
        labels_list, probs_list = model.predict(batch, k=_PREDICT_TOP_K)
        for idx, labels, probs in zip(batch_indices, labels_list, probs_list, strict=True):
            predictions[idx] = _normalize_label(labels)
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
    return predictions, spanish_probs


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
    languages, spanish_probs = _predict_languages(
        comments,
        Path(model_path),
        batch_size=batch_size,
    )
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
        ),
    )

    return output_path
