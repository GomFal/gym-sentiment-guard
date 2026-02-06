# Layer 4 — Final Model Deep Dive & Production Readiness

**Run ID**: `run.2026-01-23_005`
**Model Family**: SVC RBF

## 1. VAL vs TEST Generalization

![VAL vs TEST Comparison](figures/layer4_val_vs_test.png)

| Metric | VAL | TEST | Delta |
|--------|-----|------|-------|
| F1_neg | 0.8991 | 0.9083 | 0.0093 |
| Recall_neg | 0.9019 | 0.9102 | 0.0084 |
| Macro_F1 | 0.9343 | 0.9403 | 0.0061 |

## 2. Decision Boundary & Thresholds

![PR Curve](figures/layer4_pr_curve_neg.png)
![Threshold Curve](figures/layer4_threshold_curve.png)

**Chosen Threshold**: `0.5037`

## 3. Calibration Status

![Calibration Curve](figures/layer4_calibration_curve.png)

| Calibration Metric | Value |
|--------------------|-------|
| Brier Score | 0.0372 |
| ECE | 0.0071 |

## 4. Error Analysis (Confusion Matrices)

| VAL Confusion Matrix | TEST Confusion Matrix |
|----------------------|-----------------------|
| ![VAL CM](figures/layer4_val_cm.png) | ![TEST CM](figures/layer4_test_cm.png) |

## Conclusion

The model shows strong generalization with stable recall on TEST. The threshold is well-positioned on the PR curve to meet project constraints.