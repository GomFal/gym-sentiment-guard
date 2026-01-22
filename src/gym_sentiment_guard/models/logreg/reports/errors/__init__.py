"""
Error analysis module for post-training model inspection.

Implements ERROR_ANALYSIS_MODULE.md specification.
"""

from __future__ import annotations

from .pipeline import run_error_analysis

__all__ = ['run_error_analysis']
