"""
Main orchestrator for SVM error analysis pipeline.

Entry point that generates all artifacts, with automatic detection
of model type (Linear vs RBF) to use appropriate analysis methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sklearn.calibration import CalibratedClassifierCV

from gym_sentiment_guard.common.error_analysis import (
    build_error_table,
    compute_risk_tags,
    compute_slice_metrics,
    create_manifest,
    generate_limitations_report,
    generate_rankings,
    load_contrast_keywords,
    load_merged_data,
    load_model_bundle,
    save_error_table,
    save_limitations_report,
    save_manifest,
    save_slice_metrics,
)
from gym_sentiment_guard.utils.logging import get_logger, json_log

from .coefficients import extract_coefficients, save_coefficients
from .contributions import compute_example_contributions, save_contributions
from .support_vectors import extract_support_vector_stats, save_support_vector_stats

log = get_logger(__name__)


def _get_classifier_step(model: Any) -> tuple[str, Any]:
    """
    Get the classifier step from the model pipeline.

    Checks for common step names: 'svm', 'classifier'.

    Returns:
        Tuple of (step_name, classifier)

    Raises:
        ValueError: If no classifier step found
    """
    for step_name in ('svm', 'classifier'):
        classifier = model.named_steps.get(step_name)
        if classifier is not None:
            return step_name, classifier
    raise ValueError(
        "Model pipeline does not have 'svm' or 'classifier' step. "
        f"Available steps: {list(model.named_steps.keys())}"
    )


def _detect_svm_type(model: Any) -> str:
    """
    Detect whether the SVM is linear or RBF based on classifier attributes.

    Returns:
        'linear' if model has coef_, 'rbf' otherwise
    """
    _, classifier = _get_classifier_step(model)

    # Handle CalibratedClassifierCV wrapper
    if isinstance(classifier, CalibratedClassifierCV):
        if classifier.calibrated_classifiers_:
            base_clf = classifier.calibrated_classifiers_[0].estimator
            if hasattr(base_clf, 'coef_'):
                return 'linear'
            elif hasattr(base_clf, 'support_vectors_'):
                return 'rbf'
        return 'unknown'

    # Direct classifier
    if hasattr(classifier, 'coef_'):
        return 'linear'
    elif hasattr(classifier, 'support_vectors_'):
        return 'rbf'

    return 'unknown'


def run_error_analysis(
    config_path: Path,
    output_dir: Path,
    *,
    model_path: Path | None = None,
    predictions_path: Path | None = None,
    test_csv_path: Path | None = None,
) -> dict[str, Path]:
    """
    Run complete SVM error analysis pipeline.

    Automatically detects model type (linear/rbf) and uses appropriate methods:
    - Linear SVM: coefficients + per-example contributions
    - RBF SVM: support vector statistics

    Generates all artifacts:
    - error_table.parquet
    - ranked_errors/*.csv
    - slice_metrics.json
    - model_coefficients.json OR support_vectors.json
    - example_contributions/*.json (Linear SVM only)
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
            component='svm_error_analysis',
            config_path=str(config_path),
            output_dir=str(output_dir),
        )
    )

    artifacts: dict[str, Path] = {}

    # ==========================================================================
    # Load model and detect type
    # ==========================================================================
    model = load_model_bundle(model_path_resolved)
    svm_type = _detect_svm_type(model)

    # Extract vectorizer (SVM uses 'features' step name)
    vectorizer = model.named_steps.get('features') or model.named_steps.get('tfidf')
    if vectorizer is None:
        raise ValueError(f"No vectorizer found. Steps: {list(model.named_steps.keys())}")

    log.info(
        json_log(
            'pipeline.svm_type_detected',
            component='svm_error_analysis',
            svm_type=svm_type,
        )
    )

    # ==========================================================================
    # Load data
    # ==========================================================================
    df = load_merged_data(
        test_csv_path_resolved,
        predictions_path_resolved,
        text_column=config.get('text_column', 'comment'),
        id_column=config.get('id_column', 'id'),
    )

    # ==========================================================================
    # Build error table
    # ==========================================================================
    error_df = build_error_table(
        df,
        vectorizer,
        threshold=config['threshold'],
        low_coverage_nnz_threshold=config.get('low_coverage_nnz_threshold', 5),
        low_coverage_tfidf_threshold=config.get('low_coverage_tfidf_threshold', 0.5),
    )

    # ==========================================================================
    # Compute risk tags
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
    # Generate rankings
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
    # Slice metrics
    # ==========================================================================
    slice_metrics = compute_slice_metrics(
        error_df,
        min_slice_size=config.get('min_slice_size', 50),
    )
    slice_metrics_path = save_slice_metrics(slice_metrics, output_dir / 'slice_metrics.json')
    artifacts['slice_metrics'] = slice_metrics_path

    # ==========================================================================
    # Model interpretability (type-specific)
    # ==========================================================================
    if svm_type == 'linear':
        # Linear SVM: Extract coefficients
        try:
            coefficients = extract_coefficients(
                model,
                top_k=config.get('top_k_coefficients', 20),
            )
            coefficients_path = save_coefficients(
                coefficients, output_dir / 'model_coefficients.json'
            )
            artifacts['model_coefficients'] = coefficients_path

            # Example contributions for top errors
            contributions_examples = error_df[
                error_df['is_error']
                & (
                    (error_df['abs_margin'] >= config.get('confidence_threshold', 0.30))
                    | error_df['id'].isin(
                        error_df[error_df['is_error']].nlargest(
                            config.get('top_k_errors', 50), 'loss'
                        )['id']
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
        except ValueError as e:
            log.warning(
                json_log(
                    'pipeline.coefficients_failed',
                    component='svm_error_analysis',
                    error=str(e),
                )
            )

    elif svm_type == 'rbf':
        # RBF SVM: Extract support vector statistics
        try:
            sv_stats = extract_support_vector_stats(
                model,
                training_data=None,  # No training data access during error analysis
                n_example_svs=config.get('n_example_svs', 5),
            )
            sv_stats_path = save_support_vector_stats(
                sv_stats, output_dir / 'support_vectors.json'
            )
            artifacts['support_vectors'] = sv_stats_path
        except ValueError as e:
            log.warning(
                json_log(
                    'pipeline.support_vectors_failed',
                    component='svm_error_analysis',
                    error=str(e),
                )
            )
    else:
        log.warning(
            json_log(
                'pipeline.unknown_svm_type',
                component='svm_error_analysis',
                svm_type=svm_type,
            )
        )

    # ==========================================================================
    # KNOWN_LIMITATIONS.md
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
    # Run manifest
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
    # Add SVM-specific metadata
    manifest['svm_type'] = svm_type

    manifest_path = save_manifest(manifest, output_dir / 'run_manifest.json')
    artifacts['run_manifest'] = manifest_path

    log.info(
        json_log(
            'pipeline.completed',
            component='svm_error_analysis',
            svm_type=svm_type,
            n_artifacts=len(artifacts),
            n_samples=n_samples,
            n_errors=n_errors,
        )
    )

    return artifacts
