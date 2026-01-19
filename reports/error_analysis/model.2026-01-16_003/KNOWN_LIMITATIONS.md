# Known Model Limitations

> Auto-generated from error analysis. Do not edit manually.

---

## 1. Observed Failure Conditions

| Condition | Sample % | Error Rate | vs Overall |
|-----------|----------|------------|------------|
| **Overall** | 100% | 0.0487 | — |
| Low-coverage | 6.8% | 0.0643 | +0.0156 |
| Has-contrast | 18.1% | 0.12 | +0.0713 |

---

## 2. When NOT to Trust Predictions

The model is less reliable when:

1. **Probability is near threshold (0.37)**: 19 errors occurred in the uncertainty band.
2. **Text has few known features**: Low TF-IDF coverage indicates model blindness.
3. **Text contains contrast markers**: Mixed-sentiment reviews are harder to classify.
4. **Model is highly confident but wrong**: 43 high-confidence errors detected.

---

## 3. Suggested Escalation Rules

Consider human review or abstention when:

- `p_neg` is within ±0.10 of threshold
- Sample has < 5 non-zero TF-IDF features
- Text contains contrast markers (pero, aunque, sin embargo, etc.)
ollowing r
---

## 4. Monitoring Signals

Track in production:

1. **Near-threshold rate**: % of predictions within uncertainty band
2. **Low-coverage rate**: % of samples with sparse TF-IDF representation
3. **High-confidence error rate**: Should remain stable over time

---

## 5. Possible Future Experiments to solve this issues

1. **Contrast-aware features**: Explicit handling of "pero", "aunque" patterns
2. **Coverage-based abstention**: Reject predictions when TF-IDF sum < threshold
3. **Calibration refinement**: If ECE increases, retrain calibrator
