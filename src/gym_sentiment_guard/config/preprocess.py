"""Config models and loaders for preprocessing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PathConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    expectations_dir: Path | None = None


@dataclass(frozen=True)
class LanguageConfig:
    model_path: Path
    text_column: str = 'comment'
    batch_size: int = 512


@dataclass(frozen=True)
class CleaningConfig:
    text_column: str = 'comment'


@dataclass(frozen=True)
class DedupConfig:
    subset: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExpectationsConfig:
    required_columns: tuple[str, ...] = ('comment',)
    min_text_length: int = 1
    drop_null_comments: bool = True


@dataclass(frozen=True)
class PreprocessConfig:
    paths: PathConfig
    language: LanguageConfig
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    expectations: ExpectationsConfig = field(default_factory=ExpectationsConfig)


def load_preprocess_config(config_path: str | Path) -> PreprocessConfig:
    """Load a preprocess config YAML file."""
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f'Config file not found: {cfg_path}')

    with cfg_path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}

    base_dir = cfg_path.parent

    paths_section = data.get('paths') or {}
    language_section = data.get('language') or {}
    cleaning_section = data.get('cleaning') or {}
    dedup_section = data.get('dedup') or {}
    expectations_section = data.get('expectations') or {}

    paths = PathConfig(
        raw_dir=_resolve_path(base_dir, paths_section.get('raw_dir', 'data/raw')),
        interim_dir=_resolve_path(
            base_dir,
            paths_section.get('interim_dir', 'data/interim'),
        ),
        processed_dir=_resolve_path(
            base_dir,
            paths_section.get('processed_dir', 'data/processed'),
        ),
        expectations_dir=_resolve_optional_path(
            base_dir,
            paths_section.get('expectations_dir'),
        ),
    )

    language_model_path = language_section.get('model_path')
    if not language_model_path:
        raise ValueError('language.model_path must be set in preprocess config')

    language = LanguageConfig(
        model_path=_resolve_path(base_dir, language_model_path),
        text_column=language_section.get('text_column', 'comment'),
        batch_size=int(language_section.get('batch_size', 512)),
    )

    cleaning = CleaningConfig(
        text_column=cleaning_section.get('text_column', language.text_column),
    )

    dedup_subset = dedup_section.get('subset')
    dedup = DedupConfig(
        subset=tuple(dedup_subset) if dedup_subset else None,
    )

    expectations = ExpectationsConfig(
        required_columns=_ensure_tuple(
            expectations_section.get('required_columns', ['comment']),
        ),
        min_text_length=int(expectations_section.get('min_text_length', 1)),
        drop_null_comments=bool(expectations_section.get('drop_null_comments', True)),
    )

    return PreprocessConfig(
        paths=paths,
        language=language,
        cleaning=cleaning,
        dedup=dedup,
        expectations=expectations,
    )


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _resolve_optional_path(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return _resolve_path(base, value)


def _ensure_tuple(items: Iterable[str] | None) -> tuple[str, ...]:
    if not items:
        return tuple()
    return tuple(str(item) for item in items)
