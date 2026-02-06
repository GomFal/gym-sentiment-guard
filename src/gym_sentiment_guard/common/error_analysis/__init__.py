"""
Common error analysis module.

Shared utilities for post-training model inspection across all classifier types.
"""

from __future__ import annotations

from .error_table import build_error_table, save_error_table
from .limitations import generate_limitations_report, save_limitations_report
from .loader import load_merged_data, load_model_bundle
from .manifest import create_manifest, save_manifest
from .rankings import generate_rankings
from .risk_tags import compute_risk_tags, has_contrast_marker, load_contrast_keywords
from .slices import compute_slice_metrics, save_slice_metrics

__all__ = [
    # loader
    'load_model_bundle',
    'load_merged_data',
    # error_table
    'build_error_table',
    'save_error_table',
    # risk_tags
    'load_contrast_keywords',
    'has_contrast_marker',
    'compute_risk_tags',
    # rankings
    'generate_rankings',
    # slices
    'compute_slice_metrics',
    'save_slice_metrics',
    # limitations
    'generate_limitations_report',
    'save_limitations_report',
    # manifest
    'create_manifest',
    'save_manifest',
]

