from __future__ import annotations

import pandas as pd

from scripts.tag_ground_truth_languages import annotate_language_columns


def test_annotate_language_columns(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    es_dir = root / "es"
    en_dir = root / "en"
    es_dir.mkdir()
    en_dir.mkdir()

    spanish_csv = es_dir / "reviews_es_non_processed.csv"
    english_csv = en_dir / "reviews_en_non_processed.csv"
    pd.DataFrame({"comment": ["hola"]}).to_csv(spanish_csv, index=False)
    pd.DataFrame({"comment": ["hello"]}).to_csv(english_csv, index=False)

    updated = annotate_language_columns(root)

    es_output = spanish_csv.with_name("reviews_es.csv")
    en_output = english_csv.with_name("reviews_en.csv")
    assert es_output in updated and en_output in updated
    assert pd.read_csv(es_output)["language"].tolist() == ["es"]
    assert pd.read_csv(en_output)["language"].tolist() == ["en"]
