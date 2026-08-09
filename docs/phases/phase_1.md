# Phase 1 report

## Objective and scope

Create a modern, installable Python package skeleton without publishing it.

## Inputs received from the project owner

- `NISMO_Agent_Implementation_Plan_Phases_1_2.docx`
- `NISMO_PHASE_2_MORPH_NESTED_SAMPLING.md`
- An installed editable MorphZ 0.4.1 development package

No prior NISMO source, notebooks, tests, benchmarks, or metadata were present.

## Mathematical decisions and assumptions

The Phase 2 mathematical contract was written before its sampler. Phase 1 only
defines normalized model/proposal interfaces and explicit RNG ownership.

## Architecture/files changed

Created `pyproject.toml`, project metadata, a `src/nismo` namespace, model and
proposal protocols, immutable configuration, typed exceptions, documentation,
examples, tests, and CI.

Setuptools is used as a standards-compliant PEP 517 backend because version
69.5.1 is already available in the supplied environment and supports a clean
offline build. Project metadata remains entirely in `pyproject.toml`.

## Commands run

```bash
PYTHONPATH=src python -m pytest tests/unit/test_phase1_contracts.py
PYTHONPATH=src python examples/smoke.py
python -m build --no-isolation
python -m ruff format --check .
python -m ruff check .
python -m mypy src/nismo
python -m twine check dist/*
python -m venv --system-site-packages /tmp/nismo-phase1.23r7PJ/venv
/tmp/nismo-phase1.23r7PJ/venv/bin/python -m pip install \
  --no-deps dist/nismo-0.1.0.dev0-py3-none-any.whl
# From /tmp:
/tmp/nismo-phase1.23r7PJ/venv/bin/python -c \
  "import nismo; print(nismo.__version__); print(nismo.__file__)"
```

## Test and benchmark results

- Phase 1 unit tests: 10 passed.
- Deterministic smoke example: passed.
- Source distribution: built successfully.
- Wheel: built successfully.
- Ruff format and lint checks: passed.
- Strict mypy check of 15 source files: passed.
- Twine metadata checks for wheel and sdist: passed.
- Clean temporary-environment installation: passed.
- Import resolved to the installed wheel under the temporary environment, not
  to the repository source tree.

## Diagnostic plots

Not applicable to package infrastructure.

## Deviations from the plan

The MorphZ dependency is optional at the packaging boundary so generic
proposal tests and base imports do not pull its larger plotting/statistics
stack. Phase 2 itself uses MorphZ and tests the adapter.

## Known failures and limitations

Ruff, Twine, and mypy were not preinstalled. They were installed only into a
temporary validation environment and all required checks passed.

## Reproduction instructions

See the commands above and the repository README.

## Questions requiring owner decision

Confirm the eventual distribution name `nismo` before publication.

## Recommendation: proceed, revise, or stop

Proceed to Phase 2. Package construction, tests, formatting, lint, typing,
metadata, build, clean-wheel installation, and outside-repository import checks
pass.
