# Poetry Setup

Reviewed: 2026-05-28.

This repo is configured for Python 3.10 because `eval7==0.1.7` is not compatible with Python 3.11+ builds that moved `longintrepr.h`.

## Current Local Setup

The repo has:

- `.python-version`: `3.10.20`
- `pyproject.toml`: Poetry dependencies and Python `>=3.10,<3.11`
- `poetry.toml`: in-project virtualenvs enabled
- `poetry.lock`: locked dependencies
- `.venv/`: local virtualenv, ignored by git

## Fresh Setup

From the repo root:

```bash
pyenv install 3.10.20
pyenv local 3.10.20
make poetry-install
```

`make poetry-install` runs the extra `eval7` workaround:

```bash
poetry env use "$(pyenv which python)"
poetry run python -m pip install setuptools wheel "Cython<3"
poetry run python -m pip install --no-build-isolation eval7==0.1.7
poetry install --no-root
```

The workaround is necessary because `eval7==0.1.7` imports Cython from `setup.py`, but Poetry and modern pip build isolation do not expose the already-installed Cython package to the isolated build environment.

## Common Commands

```bash
poetry run pytest -q
poetry run python sandbox/validator.py bots/template/bot.py
poetry run python sandbox/match.py bots/template/bot.py bots/shark/bot.py --hands 20 --seed 7
poetry run python demo.py
```

Heuristic bot workflow commands:

```bash
poetry run python tools/build_heuristic_tables.py --json
poetry run python sandbox/validator.py bots/heuristic
poetry run python tools/package_heuristic.py --json
poetry run python tools/harden_submission.py --json
poetry run python tools/select_heuristic_config.py --preset candidate --progress
```

The demo listens on port 5001 by default:

```bash
DEMO_PORT=8080 poetry run python demo.py
```

## Dependency Notes

The Poetry environment pins the Docker sandbox versions for numerical and poker libraries:

- `eval7==0.1.7`
- `numpy==1.26.4`
- `scipy==1.13.0`
- `treys==0.1.8`
- `scikit-learn==1.5.2`

`flask` is included for the local demo UI. `pytest` is in the dev dependency
group for tests. `mlx` is also in the dev dependency group for Apple
Silicon-only offline training acceleration in `tools/strong_mocks/train_ppo.py`;
it is not a submitted-bot dependency and must not be imported by
`bots/heuristic/bot.py`.
