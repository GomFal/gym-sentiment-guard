"""
Artifact persistence for experiment runs.

Implements §10 of EXPERIMENT_PROTOCOL.md:
- Persist run.json with full reproducibility information
- Track validity status per §8
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .metrics import ValMetrics


@dataclass
class RunConfig:
    """Full configuration for a single experiment run."""

    # Identifiers
    run_id: str
    timestamp: str

    # Git info
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None

    # Environment
    python_version: str = ''
    sklearn_version: str = ''

    # Data paths (frozen)
    train_path: str = ''
    val_path: str = ''
    test_path: str = ''

    # TF-IDF params (§10)
    tfidf_params: dict[str, Any] = field(default_factory=dict)

    # LogReg params (§10)
    logreg_params: dict[str, Any] = field(default_factory=dict)

    # Calibration config (§10)
    calibration_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Complete result of a single experiment run."""

    config: RunConfig
    val_metrics: ValMetrics | None = None
    validity_status: Literal['valid', 'invalid', 'constraint_not_met'] = 'valid'
    invalidity_reason: str | None = None
    training_time_seconds: float = 0.0
    n_features: int = 0
    coefficient_sparsity: float = 0.0  # % non-zero coefficients


def get_git_info() -> dict[str, Any]:
    """Get current git commit, branch, and dirty status."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Check if working directory is dirty
        dirty_output = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = len(dirty_output) > 0
        return {'git_commit': commit, 'git_branch': branch, 'git_dirty': dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'git_commit': None, 'git_branch': None, 'git_dirty': None}


def get_environment_info() -> dict[str, str]:
    """Get Python and sklearn versions."""
    import sklearn

    return {
        'python_version': platform.python_version(),
        'sklearn_version': sklearn.__version__,
    }


def generate_run_id(base_dir: Path, prefix: str = 'run') -> str:
    """Generate unique run ID based on date and sequence number."""
    today = datetime.now(UTC).strftime('%Y-%m-%d')
    existing = (
        sorted(
            p.name
            for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(f'{prefix}.{today}_')
        )
        if base_dir.exists()
        else []
    )

    last_idx = int(existing[-1].split('_')[-1]) if existing else 0
    return f'{prefix}.{today}_{last_idx + 1:03d}'


def save_run_artifact(
    result: RunResult,
    output_dir: Path,
) -> Path:
    """
    Persist run.json artifact per §10.

    Args:
        result: Complete run result
        output_dir: Directory to save artifacts

    Returns:
        Path to saved run.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare run.json content
    run_data = {
        'run_id': result.config.run_id,
        'timestamp': result.config.timestamp,
        'git': {
            'commit': result.config.git_commit,
            'branch': result.config.git_branch,
            'dirty': result.config.git_dirty,
        },
        'environment': {
            'python_version': result.config.python_version,
            'sklearn_version': result.config.sklearn_version,
        },
        'data': {
            'train_path': result.config.train_path,
            'val_path': result.config.val_path,
            'test_path': result.config.test_path,
        },
        'tfidf_params': result.config.tfidf_params,
        'logreg_params': result.config.logreg_params,
        'calibration_config': result.config.calibration_config,
        'validity': {
            'status': result.validity_status,
            'reason': result.invalidity_reason,
        },
        'diagnostics': {
            'training_time_seconds': result.training_time_seconds,
            'n_features': result.n_features,
            'coefficient_sparsity': result.coefficient_sparsity,
        },
    }

    # Add VAL metrics if available
    if result.val_metrics is not None:
        run_data['val_metrics'] = asdict(result.val_metrics)
        run_data['threshold_selection'] = {
            'selected_threshold': result.val_metrics.threshold,
            'constraint_status': result.val_metrics.constraint_status,
        }

    # Save run.json
    run_path = output_dir / 'run.json'
    run_path.write_text(
        json.dumps(run_data, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    return run_path


def save_predictions(
    ids: list[str] | None,
    y_true: list[int],
    p_pos: list[float],
    p_neg: list[float],
    output_dir: Path,
    filename: str = 'val_predictions.csv',
) -> Path:
    """
    Save predictions CSV per §10 (recommended).

    Format: id, y_true, p_pos, p_neg

    Args:
        ids: Sample identifiers (optional)
        y_true: True labels
        p_pos: Predicted P(positive)
        p_neg: Predicted P(negative)
        output_dir: Directory to save
        filename: Output filename

    Returns:
        Path to saved CSV
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            'id': ids if ids is not None else list(range(len(y_true))),
            'y_true': y_true,
            'p_pos': p_pos,
            'p_neg': p_neg,
        }
    )

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path
