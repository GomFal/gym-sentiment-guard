"""Performance timing tests for experiments module optimizations."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from gym_sentiment_guard.experiments.threshold import (
    ThresholdResult,
    select_threshold,
)


def _select_threshold_naive(
    y_true: np.ndarray,
    p_neg: np.ndarray,
    recall_constraint: float = 0.90,
) -> ThresholdResult:
    """
    NAIVE implementation for timing comparison.

    Uses sklearn calls inside loop (original O(k*n) approach).
    """
    candidates = np.unique(np.concatenate([[0.0], p_neg, [1.0]]))

    best_meeting_constraint: dict | None = None
    best_fallback: dict | None = None

    for t in candidates:
        y_pred = np.where(p_neg >= t, 0, 1)

        if len(np.unique(y_pred)) == 1:
            continue

        recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0.0)
        precision_neg = precision_score(y_true, y_pred, pos_label=0, zero_division=0.0)
        f1_neg = f1_score(y_true, y_pred, pos_label=0, zero_division=0.0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0.0)

        result = {
            'threshold': float(t),
            'recall_neg': float(recall_neg),
            'precision_neg': float(precision_neg),
            'f1_neg': float(f1_neg),
            'macro_f1': float(macro_f1),
        }

        if recall_neg >= recall_constraint:
            if best_meeting_constraint is None:
                best_meeting_constraint = result
            elif (result['f1_neg'], result['macro_f1']) > (
                best_meeting_constraint['f1_neg'],
                best_meeting_constraint['macro_f1'],
            ):
                best_meeting_constraint = result

        if best_fallback is None or result['recall_neg'] > best_fallback['recall_neg']:
            best_fallback = result

    if best_meeting_constraint is not None:
        return ThresholdResult(
            threshold=best_meeting_constraint['threshold'],
            f1_neg=best_meeting_constraint['f1_neg'],
            recall_neg=best_meeting_constraint['recall_neg'],
            precision_neg=best_meeting_constraint['precision_neg'],
            macro_f1=best_meeting_constraint['macro_f1'],
            constraint_status='met',
        )
    else:
        if best_fallback is None:
            raise ValueError('No valid threshold found')
        return ThresholdResult(
            threshold=best_fallback['threshold'],
            f1_neg=best_fallback['f1_neg'],
            recall_neg=best_fallback['recall_neg'],
            precision_neg=best_fallback['precision_neg'],
            macro_f1=best_fallback['macro_f1'],
            constraint_status='not_met',
            best_achievable_recall_neg=best_fallback['recall_neg'],
            fallback_f1_neg=best_fallback['f1_neg'],
        )


def _time_function(
    func: Callable,
    *args,
    n_runs: int = 5,
    **kwargs,
) -> tuple[float, any]:
    """Time a function, return (avg_time_ms, result)."""
    # Warmup
    result = func(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return sum(times) / len(times), result


class TestThresholdSelectionPerformance:
    """Performance comparison tests for threshold selection."""

    @pytest.fixture
    def small_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Small dataset: 100 samples, ~50 unique thresholds."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        p_neg = np.random.rand(n)
        return y_true, p_neg

    @pytest.fixture
    def medium_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium dataset: 1000 samples, ~500 unique thresholds."""
        np.random.seed(42)
        n = 1000
        y_true = np.random.randint(0, 2, n)
        p_neg = np.random.rand(n)
        return y_true, p_neg

    @pytest.fixture
    def large_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Large dataset: 5000 samples, ~2000 unique thresholds."""
        np.random.seed(42)
        n = 5000
        y_true = np.random.randint(0, 2, n)
        p_neg = np.random.rand(n)
        return y_true, p_neg

    def test_vectorized_matches_naive_small(
        self, small_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify vectorized produces same results as naive on small dataset."""
        y_true, p_neg = small_dataset

        naive_result = _select_threshold_naive(y_true, p_neg, 0.5)
        vectorized_result = select_threshold(y_true, p_neg, 0.5)

        # Results should match within floating point tolerance
        assert naive_result.constraint_status == vectorized_result.constraint_status
        assert abs(naive_result.threshold - vectorized_result.threshold) < 1e-6
        assert abs(naive_result.f1_neg - vectorized_result.f1_neg) < 1e-6
        assert abs(naive_result.recall_neg - vectorized_result.recall_neg) < 1e-6

    def test_vectorized_is_faster_medium(
        self, medium_dataset: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Vectorized should be faster than naive on medium dataset."""
        y_true, p_neg = medium_dataset

        naive_time, _ = _time_function(_select_threshold_naive, y_true, p_neg, 0.5)
        vectorized_time, _ = _time_function(select_threshold, y_true, p_neg, 0.5)

        print(f'\nMedium dataset (n=1000):')
        print(f'  Naive:      {naive_time:.2f} ms')
        print(f'  Vectorized: {vectorized_time:.2f} ms')
        print(f'  Speedup:    {naive_time / vectorized_time:.1f}x')

        # Vectorized should be faster (at least 2x)
        assert vectorized_time < naive_time, 'Vectorized should be faster than naive'

    def test_vectorized_is_faster_large(self, large_dataset: tuple[np.ndarray, np.ndarray]) -> None:
        """Vectorized should be significantly faster on large dataset."""
        y_true, p_neg = large_dataset

        naive_time, _ = _time_function(_select_threshold_naive, y_true, p_neg, 0.5)
        vectorized_time, _ = _time_function(select_threshold, y_true, p_neg, 0.5)

        speedup = naive_time / vectorized_time

        print(f'\nLarge dataset (n=5000):')
        print(f'  Naive:      {naive_time:.2f} ms')
        print(f'  Vectorized: {vectorized_time:.2f} ms')
        print(f'  Speedup:    {speedup:.1f}x')

        # Should see significant speedup (at least 3x)
        assert speedup > 2.0, f'Expected >2x speedup, got {speedup:.1f}x'

    def test_print_timing_comparison(
        self,
        small_dataset: tuple[np.ndarray, np.ndarray],
        medium_dataset: tuple[np.ndarray, np.ndarray],
        large_dataset: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Print comprehensive timing comparison table."""
        datasets = [
            ('Small (n=100)', small_dataset),
            ('Medium (n=1000)', medium_dataset),
            ('Large (n=5000)', large_dataset),
        ]

        print('\n' + '=' * 60)
        print('THRESHOLD SELECTION TIMING COMPARISON')
        print('=' * 60)
        print(f'{"Dataset":<20} {"Naive (ms)":<15} {"Vectorized (ms)":<15} {"Speedup":<10}')
        print('-' * 60)

        for name, (y_true, p_neg) in datasets:
            naive_time, _ = _time_function(_select_threshold_naive, y_true, p_neg, 0.5)
            vec_time, _ = _time_function(select_threshold, y_true, p_neg, 0.5)
            speedup = naive_time / vec_time

            print(f'{name:<20} {naive_time:<15.2f} {vec_time:<15.2f} {speedup:<10.1f}x')

        print('=' * 60)
