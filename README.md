# gym-sentiment-guard
Sentiment classifier for gym reviews using TF-IDF, linear models, and reproducible preprocessing.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Preprocess CLI

Place the fastText `lid.176.bin` model under `artifacts/models/lid.176.bin` (or update the config), then run:

```bash
python -m gym_sentiment_guard.cli.main main preprocess \
  --input data/raw/Fitness_Park_Madrid_La_Vaguada.csv \
  --config configs/preprocess.yaml
```

This command (with `language.enabled: true` in `configs/preprocess.yaml`):
- Filters reviews to Spanish only via fastText.
- Normalizes text, deduplicates rows, and enforces lightweight expectations.
- Writes interim artifacts to `data/interim/` (including `<name>.non_spanish.csv` with an `es_confidence` column for rejected rows) and the final cleaned file to `data/processed/<name>.clean.csv`.

Override the final output path with `--output path/to/custom.csv` if needed.

### Skipping language filtering

Set `language.enabled: false` in `configs/preprocess.yaml` when you want to reuse the cleaning steps (expectations → normalize → dedup) without dropping non-Spanish rows—for example when preparing the multi-language evaluation datasets. In that mode, the pipeline copies the deduplicated file directly to `data/processed/<name>.clean.csv` and skips generating `.non_spanish.csv`.

## Language Evaluation Utilities

The following scripts in `scripts/` help build LID ground truth for the Google Maps evaluation dataset:

- `python scripts/tag_ground_truth_languages.py --root data/raw`
  - Walks language folders (es/pt/it/en) and stamps a `language` column in every CSV.
- `python scripts/merge_language_datasets.py --root data/raw --output-name merged.csv`
  - Concatenates each folder’s CSVs into `<folder>/merged.csv`, discarding rows whose `language` tag doesn’t match the folder name. (The language tag of each review was reviewed by Gemini 2.5 pro to assert that it matches the real language
  of the review).
- `python scripts/sample_language_reviews.py --root data/raw --samples 250`
  - Randomly samples up to 250 reviews from each `.clean.csv`, writing `<name>.sampled.csv` for eval subsets.
- `python scripts/merge_sampled_languages.py --root data/raw`
  - Combines all `.sampled.csv` files into a single `merged_sampled_ground_truth.csv` ready for LID evaluation.
- `python scripts/eval_fasttext_lid.py --data data/lid_eval/eval_dataset/merged_sampled_ground_truth.csv --model artifacts/external/models/lid.176.bin`
  - Loads the merged dataset, runs fastText predictions, prints overall accuracy plus per-language precision/recall/F1, writes `eval_results.json`, saves a confusion-matrix image (`confusion_matrix.png` by default), and reports coverage + accuracy/precision/recall/F1 for each confidence threshold (≥0.95/0.90/0.85/0.80/0.75) to design fallback policies.

## Design Decisions (LLM Chain-of-Thought)

### Language Identification Confidence Threshold

**Summary**  
FastText LID (`lid.176.bin`) was evaluated on 1000 labeled gym reviews (250 per language: en, es, pt, it). Accuracy was measured for subsets of predictions at different confidence levels.

| Confidence ≥ τ | Coverage | Accuracy |
|----------------|----------|----------|
| ≥0.95 | 78.2% | 0.997 |
| ≥0.90 | 88.2% | 0.998 |
| ≥0.85 | 91.7% | 0.998 |
| ≥0.80 | 93.4% | 0.996 |
| **≥0.75** | **94.5%** | **0.996** |
| ≥0.50 | 97.2% | 0.994 |
| ≥0.25 | 98.8% | 0.990 |

**Interpretation**
- Accuracy remains ≈0.996–0.998 for τ ≥ 0.75.  
- Most misclassifications occur below 0.75 confidence.  
- Increasing τ above 0.75 yields little accuracy gain but reduces coverage.  
- Differences between 0.996 and 0.998 accuracy are not statistically meaningful at n=1000.

**Decision**
- Use **τ = 0.75** as the confidence cutoff.  
  - If `prob ≥ 0.75`: accept fastText prediction.  
  - If `prob < 0.75`: fallback to Gemini 2.5 Flash for language verification.

This threshold balances high accuracy with minimal fallback usage.
