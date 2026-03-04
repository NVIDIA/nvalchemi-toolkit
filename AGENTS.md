# AGENTS.md — NVIDIA ALCHEMI toolkit

> Guidelines for AI coding agents operating in this repository.

## Project Overview

`nvalchemi` is an NVIDIA deep-learning framework for atomic simulations. Python 3.12+,
built with PyTorch, Pydantic, and jaxtyping. Package manager is `uv`, build backend is
`hatchling`.

## Build & Run Commands

```bash
# Install all dependencies
make install                    # or: uv sync --all-extras

# Lint (ruff, pyupgrade, whitespace, debug statements)
make lint

# Docstring coverage check
make interrogate

# Run ALL tests
make pytest                     # or: uv run coverage run -m pytest test/

# Run a SINGLE test file
uv run pytest test/data/test_data_mixin.py

# Run a SINGLE test case
uv run pytest test/data/test_data_mixin.py::TestMoveObjToDevice::test_move_tensor_to_device

# Run a single test by keyword match
uv run pytest -k "test_move_tensor" test/

# Run a specific test module (via Makefile target)
make pytest-target TARGET=test/data/test_data_mixin.py

# Specialized test suites
make pytest-data               # data, md, training, models, neighborlist
make pytest-models             # models only
make pytest-dynamics           # dynamics, md, autobatch, optim
make pytest-al                 # active learning
make pytest-utils              # utils, common, help
make pytest-training           # training, losses

# Coverage report (after running tests)
make coverage                  # fails under 75%

# License header check
make license
```

## Code Style

### Formatting & Linting

- **Formatters**: `ruff-format` (via pre-commit hooks).
- **Linter**: `ruff` with rules: `E` (pycodestyle), `F` (pyflakes), `S` (bandit),
  `I` (isort), `PERF` (performance). Only `I` rules are auto-fixable.
- **Ignored**: `E501` (line length), `S311` (random generators).
- **Per-file overrides**: `F401` ignored in `__init__.py` and `docs/*.py`;
  `S101` (assert) ignored in `test/*.py`; `E402` ignored in `examples/*.py`.
- **pyupgrade**: targets `--py310-plus`.
- **Docstrings**: `interrogate` enforces 95% coverage (excludes tests, init, magic,
  private, semiprivate, property decorators, nested functions/classes).
- **Markdown**: `markdownlint` runs in pre-commit (MD024 disabled).

### License Header

Every `.py` file MUST start with the NVIDIA SPDX license header, which
can be found in `LICENSE`. The pre-commit hooks will also run a header
check.

### Imports

- Always use `from __future__ import annotations` at the top of source files.
- Import order is enforced by ruff/isort: stdlib, third-party, local (`nvalchemi`).
- Use `TYPE_CHECKING` blocks for imports only needed at type-check time.
- Unused imports are allowed only in `__init__.py` files (F401 suppressed).

### Type Annotations

- All functions and methods MUST have type annotations.
- Use `jaxtyping` for tensor shape annotations (e.g., `Float[torch.Tensor, "V 3"]`).
- Shape dimension aliases are defined in `nvalchemi/_typing.py`:
  `B` (batch), `V` (nodes), `E` (edges), `H` (hidden), `C` (centroids), `M` (ensemble).
- Use semantic type aliases from `_typing.py` (e.g., `NodePositions`, `Forces`, `Energy`).
- Use `Annotated[type, Field(...)]` for Pydantic model fields with descriptions.
- Use `typing.Protocol` for structural typing / interfaces.
- Use `TypeAlias` for type alias declarations.

### Naming Conventions

- **Classes**: `PascalCase` — `AtomicData`, `ModelConfig`, `BaseModelMixin`.
- **Functions/methods**: `snake_case` — `compute_embeddings`, `adapt_input`.
- **Private**: prefix with `_` — `_adapt_input`, `_verify_request`, `_typing.py`.
- **Type aliases**: `PascalCase` — `NodePositions`, `GraphEmbeddings`, `ModelOutputs`.
- **Constants/module-level**: `UPPER_SNAKE_CASE` or `PascalCase` for type vars.
- **TypeVars**: single uppercase letter or short name — `T`, `F`, `C`.
- **Test classes**: `Test` prefix — `TestMoveObjToDevice`, `TestDataMixin`.
- **Test methods**: `test_` prefix with descriptive snake_case — `test_move_tensor_to_device`.

### Docstrings

- NumPy-style docstrings are required (enforced at 95% coverage).
- Must include `Parameters`, `Returns`, `Raises` sections as applicable.
- Class docstrings should include `Attributes` section.
- Use `Examples` section with doctestable code where appropriate.

### Error Handling

- Use `ValueError`, `KeyError`, `TypeError` for validation with descriptive messages.
- Use `NotImplementedError` for abstract/unimplemented methods.
- Use `warnings.warn(..., UserWarning)` for capability mismatches (not hard errors).
- Use `raise RuntimeError(...)` for internal consistency violations.
- Custom errors: `OptionalDependencyError(ImportError)` for missing optional deps.

### Pydantic Patterns

- Data structures inherit from `pydantic.BaseModel` (often mixed with custom mixins).
- Use `@model_validator(mode="after")` for cross-field consistency checks.
- Use `PlainSerializer` for custom tensor serialization.
- Use `ConfigDict(extra="allow")` when models need extensibility.
- Use `model_config = {"arbitrary_types_allowed": True}` for torch.Tensor fields.
- Heavy use of documentation with `Annotated[..., Field(description=...)]` to
document fields and intentions.

### Testing Patterns

- Framework: `pytest` with `pytest-timeout`, `pytest-asyncio`, `hypothesis`.
- Test files mirror source structure under `test/`.
- Group related tests in classes with `Test` prefix.
- Use `setup_method` for per-test fixtures within test classes.
- Use `unittest.mock.Mock`, `patch`, `patch.object` for mocking.
- Markers: `@pytest.mark.slow`, `@pytest.mark.cli`.
- Deselect slow tests with `-m 'not slow'`.
- `asyncio_mode = "auto"` — async tests run automatically.
- Test verbosity: `-vv -r xfXs` (show extra info on xfailed/xpassed/skipped).
- When possible, use `Demo*` classes; for example, `DemoModelWrapper` and
`DemoDynamics` to compose example and unit test workflows instead of
bespoke classes.

### Architecture Notes

- `nvalchemi/_typing.py`: Central type definitions — always import types from here.
- `nvalchemi/data/atomic_data.py`: Core `AtomicData` structure (Pydantic + DataMixin).
- `nvalchemi/data/batch.py`: `Batch` — batched disjoint graph (like torch_geometric).
- `nvalchemi/models/base.py`: `BaseModelMixin` — abstract interface for ML potentials.
- `nvalchemi/_imports.py`: Optional dependency management with decorator pattern.
- `nvalchemi/_utils.py`: Context managers for device/dtype/env management.

### Key Dependencies

`torch` (>=2.5.1), `pydantic` (>=2.11.7), `jaxtyping` (>=0.3.2), `loguru`,
`plum-dispatch`, `dm-tree`, `nvtx`, `numpy`, `periodictable`.
Optional: `nvidia-physicsnemo` (training extra).
