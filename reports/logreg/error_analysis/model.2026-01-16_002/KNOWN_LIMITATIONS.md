# Known Model Limitations

> Auto-generated from error analysis. Do not edit manually.

---

## 1. Observed Failure Conditions

| Condition | Sample % | Error Rate | vs Overall |
|-----------|----------|------------|------------|
| **Overall** | 100% | 0.0439 | — |
| Near-threshold | 2.6% | 0.434 | +0.3901 |
| Low-coverage | 17.7% | 0.0409 | -0.0030 |
| Has-contrast | 18.1% | 0.1013 | +0.0574 |

---

## 2. When NOT to Trust Predictions

The model is less reliable when:

1. **Probability is near threshold (0.37)**: 23 errors occurred in the uncertainty band.
2. **Text has few known features**: Low TF-IDF coverage indicates model blindness.
3. **Text contains contrast markers**: Mixed-sentiment reviews are harder to classify.
4. **Model is highly confident but wrong**: 38 high-confidence errors detected.

---

## 3. Suggested Escalation Rules

Consider human review or abstention when:

- `p_neg` is within ±0.10 of threshold
- Sample has < 5 non-zero TF-IDF features
- Text contains contrast markers (pero, aunque, sin embargo, etc.)

---

## 4. Monitoring Signals

Track in production:

1. **Near-threshold rate**: % of predictions within uncertainty band
2. **Low-coverage rate**: % of samples with sparse TF-IDF representation
3. **High-confidence error rate**: Should remain stable over time

---

## 5. Future Experiments

> Note: These should NOT be tested on the current test set.

1. **Contrast-aware features**: Explicit handling of "pero", "aunque" patterns
2. **Coverage-based abstention**: Reject predictions when TF-IDF sum < threshold
3. **Calibration refinement**: If ECE increases, retrain calibrator
