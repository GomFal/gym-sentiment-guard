"""Config models and loaders for serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    path: Path


@dataclass(frozen=True)
class ServerConfig:
    host: str = '0.0.0.0'  # noqa: S104
    port: int = 8001


@dataclass(frozen=True)
class PreprocessingConfig:
    enabled: bool = True
    structural_punctuation_path: Path | None = None


@dataclass(frozen=True)
class ValidationConfig:
    max_text_bytes: int = 51200  # 50KB
    min_text_length: int = 1


@dataclass(frozen=True)
class BatchConfig:
    max_items: int = 100
    max_text_bytes_per_item: int = 5120  # 5KB


@dataclass(frozen=True)
class LoggingConfig:
    mode: str = 'minimal'  # 'minimal' or 'requests'


@dataclass(frozen=True)
class ServingConfig:
    model: ModelConfig
    server: ServerConfig = field(default_factory=ServerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_serving_config(config_path: str | Path) -> ServingConfig:
    """Load a serving config YAML file."""
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f'Config file not found: {cfg_path}')

    with cfg_path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}

    base_dir = cfg_path.parent

    model_section = data.get('model') or {}
    server_section = data.get('server') or {}
    preprocessing_section = data.get('preprocessing') or {}
    validation_section = data.get('validation') or {}
    batch_section = data.get('batch') or {}
    logging_section = data.get('logging') or {}

    model_path = model_section.get('path')
    if not model_path:
        raise ValueError('model.path must be set in serving config')

    model = ModelConfig(
        path=_resolve_path(base_dir, model_path),
    )

    server = ServerConfig(
        host=server_section.get('host', '0.0.0.0'),  # noqa: S104
        port=int(server_section.get('port', 8001)),
    )

    preprocessing = PreprocessingConfig(
        enabled=bool(preprocessing_section.get('enabled', True)),
        structural_punctuation_path=_resolve_optional_path(
            base_dir,
            preprocessing_section.get('structural_punctuation_path'),
        ),
    )

    validation = ValidationConfig(
        max_text_bytes=int(validation_section.get('max_text_bytes', 51200)),
        min_text_length=int(validation_section.get('min_text_length', 1)),
    )

    batch = BatchConfig(
        max_items=int(batch_section.get('max_items', 100)),
        max_text_bytes_per_item=int(batch_section.get('max_text_bytes_per_item', 5120)),
    )

    logging_cfg = LoggingConfig(
        mode=logging_section.get('mode', 'minimal'),
    )

    return ServingConfig(
        model=model,
        server=server,
        preprocessing=preprocessing,
        validation=validation,
        batch=batch,
        logging=logging_cfg,
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
