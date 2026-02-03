import pytest
import pandas as pd
from pathlib import Path
from gym_sentiment_guard.models.svm.experiments.runner import SVMExperimentConfig, run_single_experiment

@pytest.fixture
def mock_data_paths(tmp_path):
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    
    # Default valid data
    pd.DataFrame({"comment": ["good"], "sentiment": ["positive"]}).to_csv(train_path, index=False)
    pd.DataFrame({"comment": ["bad"], "sentiment": ["negative"]}).to_csv(val_path, index=False)
    pd.DataFrame({"comment": ["bad"], "sentiment": ["negative"]}).to_csv(test_path, index=False)
    
    return train_path, val_path, test_path

def test_svm_runner_mapping_success(mock_data_paths):
    train_path, val_path, test_path = mock_data_paths
    
    config = SVMExperimentConfig(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        label_mapping={"positive": 1, "negative": 0},
        label_column="sentiment",
        text_column="comment",
        # Set max_iter very low to fail fast if it gets to training, 
        # but we only care about if it passes the mapping stage
        max_iter=1 
    )
    
    # We expect it might fail later due to low max_iter or small data, 
    # but it should NOT raise ValueError for unmapped labels.
    try:
        run_single_experiment(config)
    except ValueError as e:
        if "Unmapped labels" in str(e):
            pytest.fail(f"Should not have raised mapping error: {e}")
    except Exception:
        # Other errors (like ConvergenceWarning or data too small for CV) are okay here 
        # as we are testing the mapping validation block which is early in the function.
        pass

def test_svm_runner_unmapped_train(mock_data_paths, tmp_path):
    train_path, val_path, test_path = mock_data_paths
    
    # Overwrite train with invalid label
    invalid_train_path = tmp_path / "invalid_train.csv"
    pd.DataFrame({"comment": ["what"], "sentiment": ["unknown"]}).to_csv(invalid_train_path, index=False)
    
    config = SVMExperimentConfig(
        train_path=str(invalid_train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        label_mapping={"positive": 1, "negative": 0}
    )
    
    with pytest.raises(ValueError) as excinfo:
        run_single_experiment(config)
    
    assert "Unmapped labels found in TRAIN data: ['unknown']" in str(excinfo.value)
    assert f"Source: {invalid_train_path}" in str(excinfo.value)

def test_svm_runner_unmapped_val(mock_data_paths, tmp_path):
    train_path, val_path, test_path = mock_data_paths
    
    # Overwrite val with invalid label
    invalid_val_path = tmp_path / "invalid_val.csv"
    pd.DataFrame({"comment": ["oops"], "sentiment": ["buggy"]}).to_csv(invalid_val_path, index=False)
    
    config = SVMExperimentConfig(
        train_path=str(train_path),
        val_path=str(invalid_val_path),
        test_path=str(test_path),
        label_mapping={"positive": 1, "negative": 0}
    )
    
    with pytest.raises(ValueError) as excinfo:
        run_single_experiment(config)
    
    assert "Unmapped labels found in VAL data: ['buggy']" in str(excinfo.value)
    assert f"Source: {invalid_val_path}" in str(excinfo.value)
