from __future__ import annotations

import pandas as pd

from scripts.merge_language_datasets import merge_language_folders


def test_merge_language_folders(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    es_dir = root / "es"
    en_dir = root / "en"
    es_dir.mkdir()
    en_dir.mkdir()

    pd.DataFrame(
        {"comment": ["hola"], "language": ["es"]},
    ).to_csv(es_dir / "file1.csv", index=False)
    pd.DataFrame(
        {"comment": ["adios"], "language": ["pt"]},
    ).to_csv(es_dir / "file2.csv", index=False)

    pd.DataFrame(
        {"comment": ["hello"], "language": ["en"]},
    ).to_csv(en_dir / "file1.csv", index=False)
    pd.DataFrame(
        {"comment": ["hi"], "language": ["es"]},
    ).to_csv(en_dir / "file2.csv", index=False)

    merged_files = merge_language_folders(root, output_name="merged.csv")

    assert (es_dir / "merged.csv") in merged_files
    assert (en_dir / "merged.csv") in merged_files

    es_merged = pd.read_csv(es_dir / "merged.csv")
    en_merged = pd.read_csv(en_dir / "merged.csv")

    assert es_merged["language"].tolist() == ["es"]
    assert es_merged["comment"].tolist() == ["hola"]
    assert en_merged["language"].tolist() == ["en"]
    assert en_merged["comment"].tolist() == ["hello"]
