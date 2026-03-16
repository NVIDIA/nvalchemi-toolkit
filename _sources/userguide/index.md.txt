<!-- markdownlint-disable MD014 -->

(userguide)=

# User Guide

Welcome to the ALCHEMI Toolkit user guide: this side of the documentation
is to provide a high-level and conceptual understanding of the philosophy
and supported features in `nvalchemi`.

## Quick Start

The quickest way to install ALCHEMI Toolkit:

```bash
$ pip install nvalchemi-toolkit-ops
```

Make sure it is importable:

```bash
$ python -c "import nvalchemi; print(nvalchemi.__version__)"
```

## About

- [Install](about/install)
- [Introduction](about/intro)

## Core Components

- [AtomicData and Batch](data)
- [Data Loading Pipeline](datapipes)
- [Models: Wrapping ML Interatomic Potentials](models)
- [Dynamics: Optimization and MD](dynamics)

## Advanced Usage

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

about/install
about/intro
about/faq
about/contributing

```

```{toctree}
:caption: Core Components
:maxdepth: 1
:hidden:

data
datapipes
models
dynamics
```

```{toctree}
:caption: Advanced Usage
:maxdepth: 1
:hidden:

```
