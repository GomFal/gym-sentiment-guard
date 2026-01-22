"""Evaluate fastText LID accuracy on the merged evaluation dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import fasttext
import matplotlib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _predict_language_codes(
    model: fasttext.FastText._FastText,
    texts: Sequence[str],
    batch_size: int = 512,
) -> tuple[list[str], list[float]]:
    """Predict language codes and confidences using the fastText model."""
    predictions: list[str] = []
    confidences: list[float] = []
    batch: list[str] = []

    def flush_batch() -> None:
        if not batch:
            return
        labels, probs = model.predict(batch, k=1)
        for label, prob in zip(labels, probs, strict=True):
            predictions.append(label[0].replace('__label__', ''))
            confidences.append(float(prob[0]))
        batch.clear()

    for text in texts:
        normalized = text.replace('\n', ' ').strip()
        if not normalized:
            predictions.append('')
            continue
        batch.append(normalized)
        if len(batch) >= batch_size:
            flush_batch()

    flush_batch()
    return predictions, confidences


def evaluate_fasttext_lid(
    data_path: str | Path,
    model_path: str | Path,
    text_column: str = 'comment',
    label_column: str = 'language',
    batch_size: int = 512,
    output_json: str | Path | None = None,
    confusion_png: str | Path | None = None,
) -> dict:
    """Evaluate the fastText LID model and optionally write a JSON report."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    if not data_path.exists():
        raise FileNotFoundError(f'Dataset not found: {data_path}')
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    df = pd.read_csv(data_path)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f'Dataset must contain columns "{text_column}" and "{label_column}"',
        )

    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(str).str.lower().tolist()

    model = fasttext.load_model(str(model_path))
    preds, confidences = _predict_language_codes(model, texts, batch_size=batch_size)

    accuracy = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    report_text = classification_report(labels, preds, zero_division=0)
    unique_labels = sorted(set(labels) | set(preds))
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    threshold_map = {
        '0.95': 0.95,
        '0.90': 0.90,
        '0.85': 0.85,
        '0.80': 0.80,
        '0.75': 0.75,
        '0.50': 0.50,
        '0.25': 0.25,
    }

    results = {
        'dataset': str(data_path),
        'model': str(model_path),
        'accuracy': accuracy,
        'report': report,
        'report_text': report_text,
        'num_samples': len(df),
        'confidence_summary': {
            'thresholds': {
                key: {'count': int(sum(1 for c in confidences if c >= value))}
                for key, value in threshold_map.items()
            },
        },
    }
    total = len(confidences)
    for key, value in threshold_map.items():
        stats = results['confidence_summary']['thresholds'][key]
        stats['percentage'] = (stats['count'] / total * 100) if total else 0.0
        mask = [i for i, conf in enumerate(confidences) if conf >= value]
        if mask:
            subset_labels = [labels[i] for i in mask]
            subset_preds = [preds[i] for i in mask]
            acc = accuracy_score(subset_labels, subset_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                subset_labels,
                subset_preds,
                average='weighted',
                zero_division=0,
            )
            stats['metrics'] = {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'count': len(mask),
            }
        else:
            stats['metrics'] = {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'count': 0,
            }

    if confusion_png:
        confusion_path = Path(confusion_png)
        confusion_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(np.arange(len(unique_labels)))
        ax.set_yticks(np.arange(len(unique_labels)))
        ax.set_xticklabels(unique_labels)
        ax.set_yticklabels(unique_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        for i in range(len(unique_labels)):
            for j in range(len(unique_labels)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha='center',
                    va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(confusion_path, dpi=200)
        plt.close(fig)
        results['confusion_matrix_image'] = str(confusion_path)

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate fastText LID on the merged evaluation dataset.',
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/lid_eval/eval_dataset/merged_sampled_ground_truth.csv'),
        help='Path to the merged evaluation CSV.',
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('artifacts/external/models/lid.176.bin'),
        help='Path to the fastText LID model.',
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='comment',
        help='Column containing review text.',
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default='language',
        help='Column containing ground-truth language codes.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size for fastText predictions.',
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        default=Path('data/lid_eval/eval_dataset/eval_results.json'),
        help='Where to write the JSON report.',
    )
    parser.add_argument(
        '--confusion-png',
        type=Path,
        default=Path('data/lid_eval/eval_dataset/confusion_matrix.png'),
        help='Path to save the confusion matrix image.',
    )
    args = parser.parse_args()

    results = evaluate_fasttext_lid(
        data_path=args.data,
        model_path=args.model,
        text_column=args.text_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
        output_json=args.output_json,
        confusion_png=args.confusion_png,
    )

    print(f'Accuracy: {results["accuracy"]:.4f}')
    print('Classification report:')
    print(results['report_text'])
    print('Confidence coverage:')
    for threshold, stats in results['confidence_summary']['thresholds'].items():
        print(
            f'≥{threshold}: {stats["count"]} reviews ({stats["percentage"]:.1f}%)',
        )
        metrics = stats.get('metrics', {})
        if metrics and metrics['accuracy'] is not None:
            print(
                f'    accuracy={metrics["accuracy"]:.3f}, '
                f'precision={metrics["precision"]:.3f}, '
                f'recall={metrics["recall"]:.3f}, '
                f'f1={metrics["f1"]:.3f}',
            )


if __name__ == '__main__':
    main()
