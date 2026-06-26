# AGENTS.md - NVIDIA ALCHEMI Toolkit

Guidelines for AI coding agents operating in this repository.

## Project Overview

`nvalchemi-toolkit` provides the `nvalchemi` Python package: a GPU-first
framework for AI atomic simulation workflows. It covers graph-structured atomic
data, model wrappers for machine-learned interatomic potentials, batched
dynamics, hooks/reporting, and training/finetuning workflows.

- Python support: `>=3.11,<3.14`; CI and setup examples use Python 3.12.
- Package manager: `uv`; build backend: `hatchling`.
- Core dependencies: PyTorch, Pydantic v2, jaxtyping, TensorDict, Zarr, Rich,
  PhysicsNeMo, and `nvalchemi-toolkit-ops`.
- The project is in public beta. Public PRs may not be accepted immediately, but
  bug reports, feature requests, and scoped implementation discussions are welcome.

## Repository Practices

- Read `CONTRIBUTING.md` and the docs under `docs/userguide/about/` before broad
  changes; keep work tightly scoped.
- Use DCO sign-off for commits: `git commit -s -m "fix: describe change"`.
- Prefer Conventional Commits-style messages unless maintainers request otherwise.
- Install and run pre-commit hooks for development. PRs that skip pre-commit are
  not expected to be reviewed.
- The PR template expects a short description, testing notes, changelog updates,
  docstring/docs updates where applicable, and the relevant type-of-change box.

## CUDA And Environment Setup

First check CUDA availability:

```bash
nvidia-smi
```

- If `nvidia-smi` is missing or reports no usable device, use default `uv`
  commands without CUDA extras where possible.
- If it reports CUDA 12.x, pass `--extra cu12` to `uv` commands and
  `CUDA_EXTRA=cu12` to `make` targets.
- If it reports CUDA 13.x, `cu13` is the default Makefile extra; explicit
  commands can still use `--extra cu13` or `CUDA_EXTRA=cu13`.
- Do not use `uv sync --all-extras`: the CUDA variants and some model extras are
  mutually exclusive.

Common setup commands:

```bash
# Default development environment; Makefile currently defaults CUDA_EXTRA=cu13.
make install

# CUDA 12 development environment.
make install CUDA_EXTRA=cu12

# Add CUDA-aligned optional extras, for example MACE.
make install CUDA_EXTRA=cu12 OPTIONAL_EXTRAS=mace

# Direct uv equivalents.
uv sync --extra cu13
uv sync --extra cu12 --extra mace

# Include documentation dependencies when needed.
uv sync --extra cu13 --group docs
```

Optional extras include `aimnet`, `ase`, `cu12`, `cu13`, `mace`, `pymatgen`,
`tensorboard`, and `uma`. `uma` conflicts with the CUDA/MACE stack and should be
resolved in its own environment, as CI does with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-uma uv sync --extra uma --extra ase
```

## Build, Lint, Test

Use Makefile targets when possible because they keep `uv run` aligned with the
selected CUDA extra.

```bash
make lint                         # whitespace, debug, ruff check/format
make lint-fix                     # ruff check/format auto-fix path
make format                       # ruff format plus ruff check --fix
make interrogate                  # docstring coverage
make license                      # SPDX/license header validation
make docs                         # build Sphinx docs
make build                        # build package artifacts
```

Testing uses `pytest-testmon` for affected-test selection. A `.testmondata`
database is populated by full runs and reused by fast selective runs.

```bash
make test                         # affected tests with testmon --testmon-nocollect
make test-all                     # all tests and rebuild testmon database
make pytest                       # all tests with coverage, no testmon
make testmon-coverage             # CI-style testmon plus coverage

# Narrow tests with Makefile pass-through.
make test PYTEST_ARGS="test/data/test_atomic_data.py"
make pytest PYTEST_ARGS="-k test_move_tensor test/"

# Direct uv commands must include the active CUDA extra.
uv run --extra cu13 pytest test/models/test_lj_model.py
uv run --extra cu12 pytest \
  test/data/test_data_mixin.py::TestMoveObjToDevice::test_move_tensor_to_device
```

Coverage is configured in `pyproject.toml` with `fail_under = 75`, branch
coverage disabled, and `nvalchemi.coverage.xml` as the XML output. Interrogate
docstring coverage requires 95%.

## Tooling And Style

- Ruff lint rules: `E`, `F`, `S`, `I`, and `PERF`; only import sorting (`I`) is
  marked auto-fixable in `pyproject.toml`.
- Ruff ignores: `E501`, `S311`, `F722`, and `F821`.
- Per-file ignores: `F401` in `__init__.py` and `docs/*.py`; `E402` and `S101`
  in `examples/*.py`; `S101` in `test/*.py`.
- Pre-commit also runs large-file checks, trailing-whitespace, end-of-file fixer,
  YAML checks, debug-statements, Ruff, interrogate, markdownlint with `MD024`
  disabled, and the local license hook.
- Every `.py` file must start with the exact SPDX header in
  `test/_license/header.txt`.
- New source files should use `from __future__ import annotations`.
- Keep imports ordered by Ruff/isort: standard library, third-party, local
  `nvalchemi`.
- Use `TYPE_CHECKING` for type-only imports and optional-heavy imports.
- Examples in the `examples` folder should follow `sphinx-gallery` style;
this implies no interactivity, and for distributed examples they should
be skippable with the `NVALCHEMI_SPHINX_BUILD` flag (see `docs/conf.py`)

## Coding Conventions

- All public functions and methods should be type annotated and documented with
  NumPy-style docstrings.
- Use jaxtyping and semantic aliases from `nvalchemi/_typing.py` for tensor shape
  and domain types.
- Use Pydantic v2 patterns: `Annotated[..., Field(description=...)]`,
  `@model_validator(mode="after")`, `ConfigDict`, and serializers where
  appropriate.
- Prefer `typing.Protocol` for structural interfaces and `TypeAlias` for named
  aliases.
- Keep errors precise: `ValueError`, `KeyError`, or `TypeError` for validation;
  `NotImplementedError` for abstract/unimplemented behavior; `RuntimeError` for
  internal consistency failures; `warnings.warn(..., UserWarning)` for capability
  mismatches.
- Guard optional integrations with `nvalchemi._optional.OptionalDependency` and
  raise `OptionalDependencyError` through that mechanism.
- Do not add private helper functions that only wrap a single obvious call unless
  the wrapper removes real complexity or matches an existing local pattern.
- Add short comments where they explain intent or non-obvious constraints; avoid
  comments that restate the code.

## Tests

- Test files mirror package areas under `test/`: `data`, `dynamics`, `hooks`,
  `models`, and `training`.
- Test classes use `Test*`; test methods use descriptive `test_*` names.
- Use `setup_method` for per-test class fixtures when local tests already follow
  that pattern.
- Use `unittest.mock.Mock`, `patch`, and `patch.object` for mocking.
- Mark slow tests with `@pytest.mark.slow`; deselect with `-m 'not slow'`.
- CLI tests use `@pytest.mark.cli`.
- `asyncio_mode = "auto"` is enabled.
- Prefer existing demo/test utilities such as `DemoModelWrapper`, `DemoDynamics`,
  and local `conftest.py` fixtures over bespoke scaffolding.
- Add or update regression tests for behavior changes, especially model adapters,
  dynamics hooks, data serialization, training specs, and optional-dependency
  paths.

## Architecture Notes

- `nvalchemi/_typing.py`: central shape aliases and domain type aliases.
- `nvalchemi/_optional.py`: optional dependency registry and clean error path.
- `nvalchemi/_serialization.py`: tensor/model serialization helpers.
- `nvalchemi/data/`: `AtomicData`, `Batch`, data mixins, Zarr/level storage,
  datapipes, samplers, and transforms.
- `nvalchemi/models/`: `BaseModelMixin`, demo/LJ/DFTD3/Ewald/PME models,
  optional AIMNet2/MACE/UMA wrappers, neighbor filters, and composable pipelines.
- `nvalchemi/dynamics/`: base dynamics, demo dynamics, integrators, optimizers,
  sampler, sinks, hooks, and low-level ops.
- `nvalchemi/hooks/`: shared hook protocol/registry/context plus reporting,
  periodic, neighbor-list, profiling, and timing hooks.
- `nvalchemi/training/`: CLI, strategy/spec validation, runtime, distributed
  helpers, finetuning, checkpoints, losses, optimizers, and training hooks.
- `nvalchemi/distributed.py`: distributed utilities used by training and
  multi-stage workflows.

Import from concrete modules when optional exports might pull unavailable extras.
For example, prefer `from nvalchemi.models.base import BaseModelMixin` in code
that should not import optional model backends.

## Documentation And Agent Skills

- User docs live in `docs/userguide/`; API docs live in `docs/modules/`; examples
  live in `examples/`.
- Project conventions, including virial/stress/pressure signs, are documented in
  `docs/userguide/about/conventions.md`.
- Agent-facing API skills live in `.claude/skills/`. Check the relevant
  `SKILL.md` before nontrivial work in data structures/storage, dynamics,
  hooks, model wrapping, training, finetuning, reporting, losses, or Zarr
  performance.
- When docs, examples, or public APIs change, update related docs and consider
  `CHANGELOG.md` because the PR template asks for it.
