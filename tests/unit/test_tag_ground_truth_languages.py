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

    spanish_csv = es_dir / "reviews_es.csv"
    english_csv = en_dir / "reviews_en.csv"
    pd.DataFrame({"comment": ["hola"]}).to_csv(spanish_csv, index=False)
    pd.DataFrame({"comment": ["hello"]}).to_csv(english_csv, index=False)

    updated = annotate_language_columns(root)

    assert spanish_csv in updated and english_csv in updated
    assert pd.read_csv(spanish_csv)["language"].tolist() == ["es"]
    assert pd.read_csv(english_csv)["language"].tolist() == ["en"]
