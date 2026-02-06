"""
SVM error analysis module for post-training model inspection.

Supports both Linear SVM (coefficients) and RBF SVM (support vectors).
"""

from __future__ import annotations

from .pipeline import run_error_analysis

__all__ = ['run_error_analysis']
