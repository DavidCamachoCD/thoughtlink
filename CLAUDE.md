# CLAUDE.md

Instructions for Claude Code when working on the ThoughtLink project.

## Project Overview

ThoughtLink decodes non-invasive brain signals (EEG + fNIRS) into robot control commands. Built for the Global AI Hackathon (Feb 7-8, 2026), Challenge #9.

## Tech Stack

- **Python** >=3.11, <3.14
- **Package manager**: UV (not pip)
- **Build system**: Hatchling
- **ML**: scikit-learn, PyTorch, MNE-Python
- **Inference**: ONNX Runtime
- **Viz**: Streamlit, Matplotlib, Seaborn
- **Data**: HuggingFace Hub/Datasets
- **Testing**: pytest
- **Simulation**: MuJoCo (optional)

## Common Commands

```bash
# Install dependencies
uv sync

# Install with MuJoCo simulation support
uv sync --extra sim

# Run tests
uv run pytest tests/

# Train baseline models
uv run python scripts/train_baseline.py

# Train hierarchical model
uv run python scripts/train_hierarchical.py

# Benchmark latency
uv run python scripts/benchmark_latency.py
```

## Project Structure

- `src/thoughtlink/` — Main package (7 submodules: data, preprocessing, features, models, inference, bridge, viz)
- `scripts/` — Training and benchmarking entry points
- `tests/` — Unit tests (pytest)
- `configs/default.yaml` — All hyperparameters
- `notebooks/` — Jupyter notebooks for EDA and analysis

## Code Conventions

- All code, comments, docstrings, and documentation must be in **English**
- Use type hints for function signatures
- Follow existing patterns in each module (sklearn-style `fit`/`predict`/`predict_proba` interface for models)
- Config values come from `configs/default.yaml` — avoid hardcoding hyperparameters
- NumPy arrays are the standard data format between pipeline stages
- Subject-aware splitting is mandatory — never mix samples from the same subject across train/test

## Architecture Rules

- **Data flows linearly**: .npz → preprocessing → features → model → inference → bridge → robot
- **Hierarchical classifier**: Stage 1 (binary rest/active gate) must run before Stage 2 (4-class)
- **Stability pipeline order**: confidence filter → majority voting → action output
- **NIRS is secondary**: EEG is the primary signal; NIRS adds robustness but is not required for basic operation

## Testing

- Tests use synthetic data (random arrays matching real data shapes) — no dataset download needed
- Run `uv run pytest tests/` before committing
- Test files mirror source structure: `test_preprocessing.py`, `test_features.py`, `test_inference.py`

## Key Constants

- EEG: 6 channels, 500 Hz, shape per chunk: (7499, 6)
- NIRS: 40 modules, 4.76 Hz, shape per chunk: (72, 40, 3, 2, 3)
- Classes: Right Fist, Left Fist, Both Fists, Tongue Tapping, Relax
- Window: 1s (500 samples), 50% overlap
- Stimulus onset: 3s into each 15s chunk
