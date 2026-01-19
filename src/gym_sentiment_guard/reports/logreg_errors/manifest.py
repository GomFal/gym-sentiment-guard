"""
Run manifest generation for error analysis.

TASK 10: Create audit trail for reproducibility.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ...utils.logging import get_logger, json_log

log = get_logger(__name__)


def create_manifest(
    config: dict[str, Any],
    n_samples: int,
    n_errors: int,
    model_path: str,
    test_csv_path: str,
    predictions_csv_path: str,
) -> dict[str, Any]:
    """
    Create run manifest for auditing.

    Args:
        config: Full config dict used
        n_samples: Total samples analyzed
        n_errors: Total errors found
        model_path: Path to model artifact
        test_csv_path: Path to test CSV
        predictions_csv_path: Path to predictions CSV

    Returns:
        Manifest dict
    """
    manifest = {
        'timestamp': datetime.now(UTC).isoformat(),
        'config': config,
        'data': {
            'model_path': model_path,
            'test_csv_path': test_csv_path,
            'predictions_csv_path': predictions_csv_path,
            'n_samples': n_samples,
            'n_errors': n_errors,
            'error_rate': round(n_errors / n_samples, 4) if n_samples > 0 else 0.0,
        },
    }

    log.info(
        json_log(
            'manifest.created',
            component='error_analysis',
            n_samples=n_samples,
            n_errors=n_errors,
        )
    )

    return manifest


def save_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    """
    Save run manifest as JSON.

    Args:
        manifest: Manifest dict
        output_path: Path to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8',
    )

    log.info(json_log('manifest.saved', component='error_analysis', path=str(output_path)))

    return output_path
