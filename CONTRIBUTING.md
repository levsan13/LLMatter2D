# Contributing to LLMatter2D

Thank you for your interest. Bug reports, data connectors, benchmark questions and
physics improvements are all welcome.

## Reporting problems and asking questions

Open an issue at https://github.com/levsan13/LLMatter2D/issues with:

* what you ran (command or code) and the full error message;
* the versions (`llmatter2d --version`, Python, operating system);
* for wrong physics or data, the record id (`source:id`) and a reference for the
  correct value.

## Development setup

```bash
git clone https://github.com/levsan13/LLMatter2D.git
cd LLMatter2D
pip install -e ".[dev,paper]"
pytest -q
ruff check src tests
```

Tests must not need network access, an LLM server or API keys: use the offline
backend, the seed set and small synthetic inputs (see `tests/`).

## Typical contributions

* **A new data source**: subclass `llmatter2d.ingest.base.Connector`, map every value
  with its method into `Material2D`, add a `Provenance` with the reference to cite, and
  test the mapping on a small synthetic record.
* **Benchmark questions**: add lines to `benchmarks/*.jsonl` (format in
  `llmatter2d/evaluation.py`). Every question must point to the record(s) that contain
  the answer and give a tolerance for numerical answers.
* **Physics**: tight-binding extensions (longer-range hoppings, spin-orbit coupling,
  topological invariants) belong in `llmatter2d.lattices` with tests against known limits.

## Pull requests

Keep each pull request focused, add or update tests, and describe what changed and why.
By contributing you agree that your contribution is released under the MIT license.

## Conduct

Be respectful and constructive. Harassment of any kind is not tolerated.
