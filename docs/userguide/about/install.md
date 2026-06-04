<!-- markdownlint-disable MD025 MD033 MD014 -->

(install_guide)=

# Installation Guide

## Installation Methods

### From PyPI

The most straightforward way to install ALCHEMI Toolkit is via PyPI:

```bash
$ pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com \
    'nvalchemi-toolkit[cu13]'
```

For CUDA 12 environments, use the CUDA 12 PyTorch index instead:

```bash
$ pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.nvidia.com \
    'nvalchemi-toolkit[cu12]'
```

MACE support composes with the CUDA extras. Select exactly one CUDA extra
alongside `mace`:

```bash
# MACE support with the CUDA 13 stack
$ pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com \
    'nvalchemi-toolkit[cu13,mace]'

# MACE support with the CUDA 12 stack
$ pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.nvidia.com \
    'nvalchemi-toolkit[cu12,mace]'
```

```{note}
We recommend using `uv` for virtual environment, package management, and
dependency resolution. `uv` can be obtained through their installation
page found [here](https://docs.astral.sh/uv/getting-started/installation/).
```

### From Github Source

This approach is useful for obtain nightly builds by installing directly
from the source repository:

```bash
$ pip install git+https://www.github.com/NVIDIA/nvalchemi-toolkit.git
```

### Installation via `uv`

Maintainers generally use `uv`, and is the most reliable (and fastest) way
to spin up a virtual environment to use ALCHEMI Toolkit. Assuming `uv`
is in your path, here are a few ways to get started:

<details>
    <summary><b>Stable</b>, without cloning</summary>

This method is recommended for production use-cases, and when using
ALCHEMI Toolkit as a dependency for your project. The Python version
can be substituted for any other version supported by ALCHEMI Toolkit.

```bash
$ uv venv --seed --python 3.12
$ uv pip install \
    --torch-backend cu130 \
    --index https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    'nvalchemi-toolkit[cu13]'
```

For MACE and cuEquivariance support, select the matching variant:

```bash
# CUDA 13 MACE stack
$ uv pip install \
    --torch-backend cu130 \
    --index https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    'nvalchemi-toolkit[cu13,mace]'

# CUDA 12 MACE stack
$ uv pip install \
    --torch-backend cu126 \
    --index https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    'nvalchemi-toolkit[cu12,mace]'
```

</details>

<details>
    <summary><b>Nightly</b>, with cloning</summary>

This method is recommended for local development and testing.

```bash
$ git clone git@github.com:NVIDIA/nvalchemi-toolkit.git
$ cd nvalchemi-toolkit
$ uv sync --extra cu13
# include documentation tools with --group docs
```

`uv sync` creates or updates the repository `.venv`, installs the local
`nvalchemi-toolkit` package in editable mode, installs the default dependency
groups configured for the project, and uses `uv.lock` for reproducible versions.
Select exactly one CUDA extra when syncing:

```bash
# Default development stack: CUDA 13
$ uv sync --extra cu13

# CUDA 12 stack for systems that have not moved to CUDA 13 yet
$ uv sync --extra cu12

# MACE support follows the same split
$ uv sync --extra cu13 --extra mace
$ uv sync --extra cu12 --extra mace
```

The CUDA extras are intentionally mutually exclusive. Do not use
`uv sync --all-extras`, because it requests both `cu12` and `cu13` in the same
environment.

Use the same CUDA extra when running commands through `uv run`. By default,
`uv run` checks and syncs the project environment before executing the command;
bare `uv run ...` does not remember that the environment was previously synced
with `cu12`.

```bash
# Default CUDA 13 stack
$ uv run --extra cu13 pytest test/

# CUDA 12 stack
$ uv run --extra cu12 pytest test/

# CUDA 12 stack with MACE support
$ uv run --extra cu12 --extra mace pytest test/
```

The Makefile threads the selected extra through both `uv sync` and `uv run`:

```bash
# Default CUDA 13 stack
$ make test

# CUDA 12 stack
$ make test CUDA_EXTRA=cu12

# CUDA 12 stack with MACE support
$ make test CUDA_EXTRA=cu12 OPTIONAL_EXTRAS=mace
```

After a known-good sync, `uv run --no-sync ...` can run without modifying the
environment, but it also skips uv's normal environment check.

Additional dependency groups can be layered onto the selected CUDA stack:

```bash
# CUDA 13 plus documentation build dependencies
$ uv sync --extra cu13 --group docs

# Verify the environment would sync without changing it
$ uv sync --extra cu13 --dry-run

# Fail if uv.lock would need to change
$ uv sync --extra cu13 --locked
```

</details>

<details>
    <summary><b>Nightly</b>, without cloning</summary>

```{warning}
Installing nightly versions without cloning the codebase is not recommended
for production settings!
```

```bash
$ uv venv --seed --python 3.13
$ uv pip install \
    --torch-backend cu130 \
    --index https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    'nvalchemi-toolkit[cu13] @ git+https://www.github.com/NVIDIA/nvalchemi-toolkit.git'
```

</details>

<details>
    <summary>As a package dependency</summary>

To add `nvalchemi` as a dependency to your project via `uv`:

```bash
# add the last stable version
$ uv add nvalchemi
# nightly version; best practice is to pin to a version release
$ uv add "nvalchemi @ git+https://www.github.com/NVIDIA/nvalchemi-toolkit.git"
```

</details>

## Installation with Conda & Mamba

The installation procedure should be similar to other environment management tools
when using either `conda` or `mamba` managers; assuming installation from a fresh
environment:

```bash
# create a new environment named nvalchemi if needed
mamba create -n nvalchemi python=3.12 pip
mamba activate nvalchemi
pip install \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.nvidia.com \
    'nvalchemi-toolkit[cu13]'
```

## Next Steps

You should now have a local installation of `nvalchemi` ready for whatever
your use case might be! To verify, you can always run:

```bash
$ python -c "import nvalchemi; print(nvalchemi.__version__)"
```

If that doesn't resolve, make sure you've activated your virtual environment. Once
you've verified your installation, you can:

1. **Explore examples & benchmarks**: Check the `examples/` directory for tutorials
2. **Read Documentation**: Browse the user and API documentation to determine how to
integrate ALCHEMI Toolkit into your application.
