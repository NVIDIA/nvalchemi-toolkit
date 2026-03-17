<!-- markdownlint-disable MD025 MD033 MD014 -->

(install_guide)=

# Installation Guide

## Installation Methods

### From PyPI

The most straightforward way to install ALCHEMI Toolkit is via PyPI:

```bash
$ pip install nvalchemi-toolkit
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
$ uv pip install nvalchemi-toolkit
```

</details>

<details>
    <summary><b>Nightly</b>, with cloning</summary>

This method is recommended for local development and testing.

```bash
$ git clone git@github.com/NVIDIA/nvalchemi-toolkit.git
$ cd nvalchemi-toolkit
$ uv sync --all-extras
# include documentation tools with --group docs
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
$ uv pip install git+https://www.github.com/NVIDIA/nvalchemi-toolkit.git
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
pip install nvalchemi-toolkit
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
