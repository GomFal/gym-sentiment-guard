"""
Main orchestrator for error analysis pipeline.

TASK 1: Entry point that generates all artifacts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ...utils.logging import get_logger, json_log
from .coefficients import extract_coefficients, save_coefficients
from .contributions import compute_example_contributions, save_contributions
from .error_table import build_error_table, save_error_table
from .limitations import generate_limitations_report, save_limitations_report
from .loader import load_merged_data, load_model_bundle
from .manifest import create_manifest, save_manifest
from .rankings import generate_rankings
from .risk_tags import compute_risk_tags, load_contrast_keywords
from .slices import compute_slice_metrics, save_slice_metrics

log = get_logger(__name__)


def run_error_analysis(
    config_path: Path,
    output_dir: Path,
    *,
    model_path: Path | None = None,
    predictions_path: Path | None = None,
    test_csv_path: Path | None = None,
) -> dict[str, Path]:
    """
    Run complete error analysis pipeline.

    Generates all artifacts per ERROR_ANALYSIS_MODULE.md spec:
    - error_table.parquet
    - ranked_errors/*.csv
    - slice_metrics.json
    - model_coefficients.json
    - example_contributions/*.json
    - KNOWN_LIMITATIONS.md
    - run_manifest.json

    Args:
        config_path: Path to error_analysis.yaml
        output_dir: Output directory for artifacts
        model_path: Override model path from config
        predictions_path: Override predictions path from config
        test_csv_path: Override test CSV path from config

    Returns:
        Dict mapping artifact names to output paths
    """
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    # Resolve paths (CLI overrides take precedence)
    model_path_resolved = Path(model_path or config['model_path'])
    predictions_path_resolved = Path(predictions_path or config['predictions_csv_path'])
    test_csv_path_resolved = Path(test_csv_path or config['test_csv_path'])

    log.info(
        json_log(
            'pipeline.start',
            component='error_analysis',
            config_path=str(config_path),
            output_dir=str(output_dir),
        )
    )

    artifacts: dict[str, Path] = {}

    # ==========================================================================
    # TASK 2: Load model and data
    # ==========================================================================
    model = load_model_bundle(model_path_resolved)
    df = load_merged_data(
        test_csv_path_resolved,
        predictions_path_resolved,
        text_column=config.get('text_column', 'comment'),
        id_column=config.get('id_column', 'id'),
    )

    # ==========================================================================
    # TASK 3: Build error table
    # ==========================================================================
    error_df = build_error_table(
        df,
        model,
        threshold=config['threshold'],
        low_coverage_nnz_threshold=config.get('low_coverage_nnz_threshold', 5),
        low_coverage_tfidf_threshold=config.get('low_coverage_tfidf_threshold', 0.5),
    )

    # ==========================================================================
    # TASK 4: Compute risk tags
    # ==========================================================================
    contrast_keywords_path = Path(
        config.get('contrast_keywords_path', 'configs/contrast_keywords.txt')
    )
    contrast_keywords = load_contrast_keywords(contrast_keywords_path)

    error_df = compute_risk_tags(
        error_df,
        threshold=config['threshold'],
        near_threshold_band=config.get('near_threshold_band', 0.10),
        contrast_keywords=contrast_keywords,
    )

    # Save error table
    error_table_path = save_error_table(error_df, output_dir / 'error_table.parquet')
    artifacts['error_table'] = error_table_path

    # ==========================================================================
    # TASK 5: Generate rankings
    # ==========================================================================
    rankings_dir = output_dir / 'ranked_errors'
    ranking_paths = generate_rankings(
        error_df,
        rankings_dir,
        confidence_threshold=config.get('confidence_threshold', 0.30),
        top_k=config.get('top_k_errors', 50),
    )
    artifacts.update({f'ranking_{k}': v for k, v in ranking_paths.items()})

    # ==========================================================================
    # TASK 6: Slice metrics
    # ==========================================================================
    slice_metrics = compute_slice_metrics(
        error_df,
        min_slice_size=config.get('min_slice_size', 50),
    )
    slice_metrics_path = save_slice_metrics(slice_metrics, output_dir / 'slice_metrics.json')
    artifacts['slice_metrics'] = slice_metrics_path

    # ==========================================================================
    # TASK 7: Model coefficients
    # ==========================================================================
    coefficients = extract_coefficients(
        model,
        top_k=config.get('top_k_coefficients', 20),
    )
    coefficients_path = save_coefficients(coefficients, output_dir / 'model_coefficients.json')
    artifacts['model_coefficients'] = coefficients_path

    # ==========================================================================
    # TASK 8: Example contributions (for top errors only)
    # ==========================================================================
    # Get examples from high-confidence and top-loss rankings
    contributions_examples = error_df[
        error_df['is_error']
        & (
            (error_df['abs_margin'] >= config.get('confidence_threshold', 0.30))
            | error_df['id'].isin(
                error_df[error_df['is_error']].nlargest(config.get('top_k_errors', 50), 'loss')[
                    'id'
                ]
            )
        )
    ].head(config.get('top_k_errors', 50))

    if len(contributions_examples) > 0:
        contributions = compute_example_contributions(
            contributions_examples,
            model,
            top_k=10,
        )
        contributions_dir = output_dir / 'example_contributions'
        save_contributions(contributions, contributions_dir)
        artifacts['example_contributions'] = contributions_dir

    # ==========================================================================
    # TASK 9: KNOWN_LIMITATIONS.md
    # ==========================================================================
    n_high_conf = len(
        error_df[
            error_df['is_error']
            & (error_df['abs_margin'] >= config.get('confidence_threshold', 0.30))
        ]
    )
    n_near_thresh = len(error_df[error_df['is_error'] & error_df['near_threshold']])

    limitations_content = generate_limitations_report(
        slice_metrics,
        n_high_confidence_errors=n_high_conf,
        n_near_threshold_errors=n_near_thresh,
        threshold=config['threshold'],
    )
    limitations_path = save_limitations_report(
        limitations_content, output_dir / 'KNOWN_LIMITATIONS.md'
    )
    artifacts['known_limitations'] = limitations_path

    # ==========================================================================
    # TASK 10: Run manifest
    # ==========================================================================
    n_samples = len(error_df)
    n_errors = int(error_df['is_error'].sum())

    manifest = create_manifest(
        config=config,
        n_samples=n_samples,
        n_errors=n_errors,
        model_path=str(model_path_resolved),
        test_csv_path=str(test_csv_path_resolved),
        predictions_csv_path=str(predictions_path_resolved),
    )
    manifest_path = save_manifest(manifest, output_dir / 'run_manifest.json')
    artifacts['run_manifest'] = manifest_path

    log.info(
        json_log(
            'pipeline.completed',
            component='error_analysis',
            n_artifacts=len(artifacts),
            n_samples=n_samples,
            n_errors=n_errors,
        )
    )

    return artifacts
