# HTM Test Suite

This directory contains pytest-based tests for encoder behavior and real-data learning dynamics.

## Running the tests

From the repository root:

```bash
pytest
```

To run a single file:

```bash
pytest tests/test_rdse.py
```

## What's covered

Current tests focus on:

- **RDSE encoder** (Random Distributed Scalar Encoder)
- **DateEncoder** edge cases and configuration validation
- **Real-data behavior** of `ColumnField` on a sine-driven sequence

Pytest collects both pytest-style tests and `unittest.TestCase` classes.

## Requirements

- Python 3.13+
- NumPy
- pytest

Install dependencies:

```bash
pip install -r requirements.txt
```

## Adding new tests

- Prefer pytest-style tests in new files under [tests](tests).
- Name tests `test_*.py` and functions `test_*` so pytest can discover them.

Example:

```python
def test_my_feature():
    assert 1 + 1 == 2
```

## Notes

Some tests (e.g., [tests/test_real_data.py](tests/test_real_data.py)) run longer loops. If you need a quicker run, execute only the encoder tests:

```bash
pytest tests/test_rdse.py tests/test_date_encoder.py
```
