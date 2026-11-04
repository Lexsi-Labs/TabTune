# Contributing to TabTune

Thanks for considering it. Bug reports, model integrations, documentation fixes and
benchmark results are all useful. This page is the whole process.

## Set up

```bash
git clone https://github.com/<you>/TabTune.git
cd TabTune
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev,docs]"
pre-commit install
```

Run the suite before you start, so you know what green looks like on your machine:

```bash
pytest -q                 # about 1,600 tests, CPU only, no downloads
pytest tests/test_registry.py tests/test_imports.py   # the fast ones
```

Nothing in `tests/` downloads weights. Tests that need a real checkpoint are marked and
skip when it is absent.

## Code

- `ruff`, `black` and `isort` run through pre-commit; the CI runs them too.
- Type hints on public functions. Google-style docstrings; the API reference is rendered
  from them.
- Small pull requests with one purpose. A model integration and a refactor of the
  processor are two pull requests.
- Vendored model code under `tabtune/models/<name>/` is not edited. If upstream has a bug,
  work around it in `tabtune_support.py` and link the upstream issue.
- A change to public behaviour comes with a test and a line in `CHANGELOG.md` under
  *Unreleased*.

## Adding a model

1. Vendor the upstream tree under `tabtune/models/<name>/`, keep its `LICENSE`, rewrite
   its absolute imports to the vendored path, and change nothing else.
2. Write `tabtune/models/<name>/tabtune_support.py`: an estimator that accepts TabTune's
   constructor conventions (`device`, `model_path`, `tuning_strategy`), exposes the torch
   module for LoRA, and, if the model fine-tunes, the episode loss and validation hooks the
   episodic driver needs.
3. Add a preprocessor in `tabtune/Dataprocess/` only if the default is wrong for the
   model. Models that preprocess internally get a pass-through.
4. Register a `ModelSpec` in `tabtune/registry/catalog.py`: tasks, strategies, modes, the
   capability envelope, the weight licence with its URL, and the paper.
5. Add the dispatch branch in `tabtune/TuningManager/tuning.py` and, if LoRA applies, an
   entry in `MODEL_LORA_TARGETS` in `peft_utils.py`.
6. Add `docs/models/<name>.md` (copy the structure of an existing page) and a row in the
   nav in `mkdocs.yml`.
7. Add `tests/test_<name>_integration.py` that runs inference, fine-tuning and LoRA on a
   tiny randomly initialised network, and bump the registry count in
   `tests/test_registry.py` and `tests/test_TimeSeries_registry.py`.

## Documentation

```bash
pip install -r requirements-docs.txt
mkdocs serve            # http://127.0.0.1:8000
mkdocs build --strict   # what the CI runs
```

Pages are Markdown under `docs/`, built with mkdocs-material. Sentence case for headings,
runnable examples, no claims the code does not back. If you change a public signature,
change the page that documents it in the same pull request.

## Pull requests

- Open an issue first for anything larger than a fix, so the design is agreed before the
  code exists.
- Describe what changed and why, and paste the test output.
- One approving review and green CI merge it. Maintainers may squash.

## Reporting

- Bugs and feature requests: [GitHub Issues](https://github.com/Lexsi-Labs/TabTune/issues).
- Questions: [Discord](https://discord.gg/dSB62Q7A).
- Security problems: see `SECURITY.md`. Please do not file them as public issues.

By contributing you agree that your contribution is licensed under the MIT License that
covers the project. Vendored model code keeps its own licence.
