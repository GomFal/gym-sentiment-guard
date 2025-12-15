## Contributing to gym-sentiment-guard

Thanks for helping improve this project! Please follow the guidelines below to keep contributions smooth.

### Quick start

1. Use Python 3.11+ (recommend pyenv or system install).
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   . .venv/Scripts/activate  # Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Run tests before pushing:
   ```bash
   pytest
   ```

### Development workflow

- Fork the repo and work on feature branches: `feat/...`, `fix/...`, `docs/...`, `chore/...`.
- Always open PRs against `main`; do not push directly.
- Link issues in PR descriptions using “Fixes #123” or “Closes #123”.

### Code quality

- Lint:
  ```bash
  ruff check .
  ```
- Format (if needed):
  ```bash
  ruff format .
  ```
- Keep imports and typing clean; follow existing style.

### Testing guidelines

- Unit tests live under `tests/unit/`.
- Add or update tests for every bug fix and feature.
- Use fixtures under `tests/data/fixtures/` when sample data is necessary.

### ML / data notes

- Do **not** commit large datasets or proprietary data.
- Store small sample fixtures only in `tests/data/fixtures/`.
- Log experiment configs/metrics under `artifacts/` so runs are reproducible.

### Definition of Done

- CI and local tests pass.
- Code/lint checks pass (`ruff check`, `ruff format` if used).
- Docs/configs updated where applicable.
- Acceptance criteria met and reproducible command listed in the PR.
- No stray data files committed; changes are review-ready.
