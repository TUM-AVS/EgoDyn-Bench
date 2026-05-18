# Testing and CI/CD

## Test suites

All tests run with `pytest` and require no dataset download -- they use synthetic data.

| Suite | Path | Scope |
|-------|------|-------|
| Dynamics features | `dataset/tests/test_dynamics_features.py` | Speed, acceleration, yaw-rate, jerk, smoothing, validation |
| QA generation | `dataset/tests/test_qa_generation.py` | All 12 labeling rules, config validation, end-to-end generator |
| Answer parsers | `tests/test_parsers.py` | Text normalization, binary/multiclass/numeric parsing |
| Evaluation metrics | `tests/test_metrics.py` | Accuracy, F1, confusion matrix, consistency, full `evaluate()` pipeline |

## Running tests locally

```bash
conda activate dynamics-benchmark

# Run all tests
python -m pytest tests/ dataset/tests/ -v

# Run a single suite
python -m pytest tests/test_metrics.py -v

# Run a single test
python -m pytest tests/test_metrics.py::TestSemanticAccuracy::test_perfect -v
```

**Without conda activate** (e.g. from scripts or cron):

```bash
conda run -n dynamics-benchmark python -m pytest tests/ dataset/tests/ -v
```

## GitLab CI

Tests run automatically on every push via `.gitlab-ci.yml`:

```yaml
unit_tests:
  stage: test
  image: python:3.11-slim
  cache:
    paths:
      - .pip-cache/
  before_script:
    - pip install pytest numpy scipy pyyaml
  script:
    - python -m pytest tests/ -v
    - python -m pytest dataset/tests/ -v
```

The pipeline also runs **GitLab Secret Detection** to prevent accidental credential commits.

### CI dependencies

The CI job installs only the minimal packages needed to run tests (`pytest`, `numpy`, `scipy`, `pyyaml`). Heavyweight dependencies like `nuscenes-devkit`, `openai`, and `matplotlib` are not needed for unit tests.

### Adding new tests

1. Place test files in `tests/` (evaluation module) or `dataset/tests/` (dataset module)
2. Name files `test_*.py` and classes `Test*`
3. The CI pipeline picks them up automatically

## Current test count

111 tests across all suites (as of February 2026).
