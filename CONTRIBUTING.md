# Contributing to ThoughtLink

## Getting Started

### Prerequisites

- Python >=3.11, <3.14
- [UV](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone <repo-url>
cd hackaton
uv sync
```

To enable MuJoCo robot simulation:

```bash
uv sync --extra sim
```

## Development Workflow

### Branching

- `main` — stable, working code
- Feature branches: `feature/<short-description>`
- Bug fixes: `fix/<short-description>`

```bash
git checkout -b feature/my-feature
```

### Running Tests

Always run tests before pushing:

```bash
uv run pytest tests/
```

Tests use synthetic data and do not require downloading the dataset.

### Code Style

- **Language**: All code, comments, docstrings, and docs must be in English
- **Type hints**: Use them on function signatures
- **Imports**: Standard library first, third-party second, local third
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Hyperparameters**: Read from `configs/default.yaml`, do not hardcode

### Commit Messages

Use clear, concise commit messages:

```
Add band power feature extraction
Fix windowing overlap calculation
Update NIRS baseline correction logic
```

Prefix with the area when helpful:

```
data: add subject-aware cross-validation folds
models: implement hierarchical 2-stage classifier
inference: add hysteresis to confidence filter
```

## Project Modules

| Module | Purpose |
|--------|---------|
| `data/` | Dataset loading and splitting |
| `preprocessing/` | EEG/NIRS signal cleaning and windowing |
| `features/` | Feature extraction and fusion |
| `models/` | ML classifiers (sklearn, PyTorch) |
| `inference/` | Real-time decoding and stability |
| `bridge/` | Intent-to-action mapping and robot control |
| `viz/` | Dashboard and visualizations |

### Adding a New Model

1. Create a file in `src/thoughtlink/models/`
2. Implement the sklearn interface: `fit(X, y)`, `predict(X)`, `predict_proba(X)`
3. Add it to the training script in `scripts/`
4. Add tests in `tests/`

### Adding New Features

1. Add extraction logic in `src/thoughtlink/features/`
2. Update `fusion.py` if the feature should be included in the combined vector
3. Add tests verifying output shape and value ranges

## Data Handling

- Dataset: [KernelCo/robot_control](https://huggingface.co/datasets/KernelCo/robot_control) on HuggingFace
- **Never commit data files** (`.npz`, `.pkl`, model weights) to the repo
- Always split by `subject_id` to prevent data leakage
- The `.gitignore` should exclude `data/`, `results/`, and model artifacts

## Team

- **David** — Infrastructure, pipeline, integration
- **Nat** — Signal processing, models, metrics
