# Contributing

Contributions that improve reproducibility, evaluation correctness, CPU
efficiency, or documentation are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make test
make paper
```

Experiment dependencies can be added with `pip install -e ".[experiments,dev]"`.
Do not commit RDD2022 files, model weights, generated predictions, dependency
directories, or other large outputs.

## Pull requests

1. Open an issue first for changes to the frozen scientific protocol or reported
   results.
2. Keep unrelated changes separate.
3. Add or update tests for behavioral changes.
4. Run `python -m ruff check .`, `make test`, and `make paper`.
5. State whether results, hashes, licensing, or data handling are affected.

Test-set metrics must never guide model or hyperparameter selection. New result
claims require a command, configuration, environment description, and artifact
hashes.
