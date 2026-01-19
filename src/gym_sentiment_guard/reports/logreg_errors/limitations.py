"""
KNOWN_LIMITATIONS.md generation for error analysis.

TASK 9: Auto-generate deployment knowledge from analysis results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...utils.logging import get_logger, json_log

log = get_logger(__name__)


def generate_limitations_report(
    slice_metrics: dict[str, Any],
    n_high_confidence_errors: int,
    n_near_threshold_errors: int,
    threshold: float,
) -> str:
    """
    Generate KNOWN_LIMITATIONS.md content.

    Args:
        slice_metrics: Slice metrics dict
        n_high_confidence_errors: Count of high-confidence wrong
        n_near_threshold_errors: Count of near-threshold wrong
        threshold: Decision threshold

    Returns:
        Markdown content string
    """
    overall = slice_metrics.get('overall', {})
    near_threshold = slice_metrics.get('near_threshold', {})
    low_coverage = slice_metrics.get('low_coverage', {})
    has_contrast = slice_metrics.get('has_contrast', {})

    content = f"""# Known Model Limitations

> Auto-generated from error analysis. Do not edit manually.

---

## 1. Observed Failure Conditions

| Condition | Sample % | Error Rate | vs Overall |
|-----------|----------|------------|------------|
| **Overall** | 100% | {overall.get('error_rate', 'N/A')} | — |
"""

    # Add slice rows if not skipped
    if not near_threshold.get('skipped', True):
        overall_n = overall.get('n', 1)
        sample_pct = round(near_threshold['n'] / overall_n * 100, 1)
        delta = round(near_threshold['error_rate'] - overall.get('error_rate', 0), 4)
        content += (
            f'| Near-threshold | {sample_pct}% | {near_threshold["error_rate"]} | {delta:+.4f} |\n'
        )

    if not low_coverage.get('skipped', True):
        overall_n = overall.get('n', 1)
        sample_pct = round(low_coverage['n'] / overall_n * 100, 1)
        delta = round(low_coverage['error_rate'] - overall.get('error_rate', 0), 4)
        content += (
            f'| Low-coverage | {sample_pct}% | {low_coverage["error_rate"]} | {delta:+.4f} |\n'
        )

    if not has_contrast.get('skipped', True):
        overall_n = overall.get('n', 1)
        sample_pct = round(has_contrast['n'] / overall_n * 100, 1)
        delta = round(has_contrast['error_rate'] - overall.get('error_rate', 0), 4)
        content += (
            f'| Has-contrast | {sample_pct}% | {has_contrast["error_rate"]} | {delta:+.4f} |\n'
        )

    content += f"""
---

## 2. When NOT to Trust Predictions

The model is less reliable when:

1. **Probability is near threshold ({threshold:.2f})**: {n_near_threshold_errors} errors occurred in the uncertainty band.
2. **Text has few known features**: Low TF-IDF coverage indicates model blindness.
3. **Text contains contrast markers**: Mixed-sentiment reviews are harder to classify.
4. **Model is highly confident but wrong**: {n_high_confidence_errors} high-confidence errors detected.

---

## 3. Suggested Escalation Rules

Consider human review or abstention when:

- `p_neg` is within ±0.10 of threshold
- Sample has < 5 non-zero TF-IDF features
- Text contains contrast markers (pero, aunque, sin embargo, etc.)

---

## 4. Monitoring Signals

Track in production:

1. **Near-threshold rate**: % of predictions within uncertainty band
2. **Low-coverage rate**: % of samples with sparse TF-IDF representation
3. **High-confidence error rate**: Should remain stable over time

---

## 5. Future Experiments

> Note: These should NOT be tested on the current test set.

1. **Contrast-aware features**: Explicit handling of "pero", "aunque" patterns
2. **Coverage-based abstention**: Reject predictions when TF-IDF sum < threshold
3. **Calibration refinement**: If ECE increases, retrain calibrator
"""

    return content


def save_limitations_report(content: str, output_path: Path) -> Path:
    """
    Save KNOWN_LIMITATIONS.md.

    Args:
        content: Markdown content
        output_path: Path to save

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')

    log.info(json_log('limitations.saved', component='error_analysis', path=str(output_path)))

    return output_path
