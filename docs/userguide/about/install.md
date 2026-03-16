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
$ uv sync
```

</details>

<details>
    <summary><b>Nightly</b>, without cloning</summary>

```{warning}
Installing nightly versions without cloning the codebase is not recommended
for production settings!
```

```bash
$ uv venv --seed --python 3.12
$ uv pip install git+https://www.github.com/NVIDIA/nvalchemi-toolkit.git
```

</details>

Includes Sphinx and related tools for building documentation.

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

## Docker Usage

Given the modular nature of `nvalchemi`, we do not provide a base Docker image.
Instead, the snippet below is a suggested base image that follows the requirements
of NVIDIA `warp-lang`, and installs `uv` for Python management:

```docker
# uses a lightweight Ubuntu-based image with CUDA 13
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# grab package updates and other system dependencies here
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
# copy uv for venv management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv venv --seed --python 3.12 /opt/venv
# this sets the default virtual environment to use
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# install ALCHEMI Toolkit
RUN uv pip install nvalchemi-toolkit
```

This image can potentially be used as a basis for your application and/or development
environment. Your host system should have the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
installed, and at runtime, include `--gpus all` as a flag to container run statements to
ensure that GPUs are exposed to the container.

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
