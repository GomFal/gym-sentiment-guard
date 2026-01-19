"""
Evaluate DistilBERT multilingual sentiment model on frozen test set.

This script runs inference on the full test set using a HuggingFace transformer
model that outputs 3 classes (positive, negative, neutral). Since our test set
only has binary labels, neutral predictions are exported to a separate CSV
for manual analysis.

Usage:
    python scripts/eval_transformer_sentiment.py
"""
from pathlib import Path
import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import pipeline


# --- Configuration ---
TEST_CSV = Path("data/frozen/sentiment_logreg/2025.12.15_01/test/test.csv")
NEUTRAL_OUTPUT = Path("scripts/neutral_predictions.csv")
INCORRECT_OUTPUT = Path("scripts/incorrect_predictions.csv")
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
BATCH_SIZE = 32
TEXT_COL = "comment"
LABEL_COL = "sentiment"


def main():
    print(f"Loading test data from {TEST_CSV}...")
    df = pd.read_csv(TEST_CSV)
    texts = df[TEXT_COL].tolist()
    n_samples = len(texts)
    print(f"Loaded {n_samples} samples")

    # Initialize model (device=-1 for CPU, device=0 for GPU)
    print(f"\nLoading model: {MODEL_NAME}...")
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1,  # Use CPU; change to 0 for CUDA GPU
    )
    print("Model loaded successfully")

    # Run inference in batches
    print(f"\nRunning inference (batch_size={BATCH_SIZE})...")
    predictions = []
    start_time = time.time()

    for i in range(0, n_samples, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results = classifier(batch, truncation=True, max_length=512)
        predictions.extend(results)

        # Progress indicator
        processed = min(i + BATCH_SIZE, n_samples)
        if processed % 200 == 0 or processed == n_samples:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed}/{n_samples} ({rate:.1f} samples/sec)")

    elapsed_total = time.time() - start_time
    print(f"Inference completed in {elapsed_total:.1f}s")

    # Process predictions
    df["pred_label"] = [p["label"] for p in predictions]
    df["pred_score"] = [p["score"] for p in predictions]

    # Split neutrals for separate analysis
    neutral_mask = df["pred_label"] == "neutral"
    n_neutrals = neutral_mask.sum()

    if n_neutrals > 0:
        neutrals_df = df[neutral_mask][
            ["id", TEXT_COL, LABEL_COL, "pred_label", "pred_score"]
        ]
        neutrals_df.to_csv(NEUTRAL_OUTPUT, index=False)
        print(f"\nNeutral predictions ({n_neutrals}) saved to: {NEUTRAL_OUTPUT}")

    # Compute metrics on non-neutral samples
    eval_df = df[~neutral_mask].copy()
    n_eval = len(eval_df)

    print(f"\n{'='*50}")
    print("TRANSFORMER SENTIMENT EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Total samples:      {n_samples}")
    print(f"Neutral predictions: {n_neutrals} ({100*n_neutrals/n_samples:.1f}%)")
    print(f"Evaluated samples:   {n_eval} ({100*n_eval/n_samples:.1f}%)")

    if n_eval > 0:
        y_true = eval_df[LABEL_COL]
        y_pred = eval_df["pred_label"]

        # Overall accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy (excl. neutrals): {acc:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))

        # Confusion matrix
        print("Confusion Matrix:")
        print("                 Predicted")
        print("              neg      pos")
        cm = confusion_matrix(y_true, y_pred, labels=["negative", "positive"])
        print(f"Actual neg   {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"       pos   {cm[1,0]:5d}    {cm[1,1]:5d}")

        # Per-class breakdown for neutrals
        if n_neutrals > 0:
            print(f"\nNeutral predictions breakdown (by true label):")
            neutral_by_true = df[neutral_mask][LABEL_COL].value_counts()
            for label, count in neutral_by_true.items():
                pct = 100 * count / n_neutrals
                print(f"  True {label}: {count} ({pct:.1f}%)")

        # Export incorrectly predicted reviews
        incorrect_mask = y_true != y_pred
        n_incorrect = incorrect_mask.sum()
        if n_incorrect > 0:
            incorrect_df = eval_df[incorrect_mask][
                ["id", TEXT_COL, LABEL_COL, "pred_label", "pred_score"]
            ].copy()
            incorrect_df.columns = ["id", "text", "true_label", "pred_label", "confidence"]
            incorrect_df.to_csv(INCORRECT_OUTPUT, index=False)
            print(f"\nIncorrect predictions ({n_incorrect}) saved to: {INCORRECT_OUTPUT}")
    else:
        print("\nNo non-neutral predictions to evaluate!")


if __name__ == "__main__":
    main()
