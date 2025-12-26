# /predict/explain Endpoint Implementation Plan

## 1. Objective
Implement a new API endpoint `/predict/explain` that provides local interpretability for sentiment predictions. For each input text, return the top contributing words (features) and their impact scores based on the trained Logistic Regression model.

## 2. Technical Architecture

### 2.1. Feature Importance Algorithm (White-Box)
Since the model is a `Pipeline(TfidfVectorizer -> LogisticRegression)`, we can calculate exact feature contributions without approximation (like LIME or SHAP).

**Formula:**
$$ \text{Importance}(w) = \text{TFIDF}(w) \times \beta_w $$
Where:
*   $\text{TFIDF}(w)$ is the specific weight of word $w$ in the input text.
*   $\beta_w$ is the learned coefficient for word $w$ from the Logistic Regression.

**Step-by-Step Logic:**
1.  **Access Pipeline Components**:
    *   `vectorizer = artifact.model.steps[0][1]`
    *   `classifier = artifact.model.steps[-1][1]`
2.  **Access Global State**:
    *   `feature_names = vectorizer.get_feature_names_out()` (Vocabulary array)
    *   `coefficients = classifier.coef_[0]` (Dense array of weights)
3.  **Process Input (Per Text)**:
    *   `sparse_vector = vectorizer.transform([text])`
    *   Iterate over non-zero indices of the sparse vector: `indices = sparse_vector.indices`
    *   For each index `idx`:
        *   `word = feature_names[idx]`
        *   `tfidf_value = sparse_vector[0, idx]`
        *   `coef = coefficients[idx]`
        *   `score = tfidf_value * coef`
4.  **Ranking**:
    *   Sort features by `abs(score)` descending.
    *   Select top K (e.g., 10).

### 2.2. Schema Design (`src/gym_sentiment_guard/serving/schemas.py`)

New data models required for the response structure.

```python
class FeatureImportance(BaseModel):
    feature: str      # The word/token (e.g., "excelente")
    importance: float # The signed contribution (e.g., 2.45)

class ExplainResponse(PredictResponse):
    explanation: list[FeatureImportance] # The list of features
```

The endpoint will return `list[ExplainResponse]`.

### 2.3. Core Logic (`src/gym_sentiment_guard/serving/predict.py`)

A new function `explain_predictions` will be added.

**Signature:**
```python
def explain_predictions(
    texts: list[str],
    artifact: ModelArtifact,
    apply_preprocessing: bool = True,
    structural_punctuation: str | None = None,
    top_k: int = 10
) -> list[ExplainResponse]:
```

**Memory Optimization Note:**
The `feature_names` array can be large. Accessing it is cheap, but we should strictly avoid copying the entire dense coefficient array per request. We only access the indices present in the input text.

### 2.4. API Endpoint (`src/gym_sentiment_guard/serving/app.py`)

**Route:** `POST /predict/explain`

**Flow:**
1.  **Validate**: Reuse `_validate_text_size` and batch limits.
2.  **Execute**: Call `explain_predictions`.
3.  **Return**: List of explained predictions.

**Tags**: `['Prediction', 'Explainability']`
