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
- [Conventions](about/conventions)

## Core Components

- [AtomicData and Batch](data)
- [Data Loading Pipeline](datapipes)
- {doc}`Models: Wrapping ML Interatomic Potentials <models>`
- {doc}`Training: Strategy and Runtime <training>`
- {doc}`Losses: Composable Training Terms <losses>`
- {doc}`Fine-Tuning Pretrained Models <finetuning>`
- {doc}`Hooks: Observe & Modify <hooks>`
- {doc}`Reporting: Summaries and Dashboards <reporting>`
- [Dynamics: Optimization and MD](dynamics)

## Distributed Simulations

- {doc}`Overview: Domain Decomposition <distributed>`
- {doc}`ShardTensor: Per-Atom Fields Across Ranks <distributed_shardtensor>`
- {doc}`Bring Your Own Model: Authoring a Spec <distributed_byo>`
- {doc}`Architecture & design (deep dive) <distributed_design>`

## Advanced Usage

- [Distributed Training](distributed_training)
- [Zarr Compression Tuning](zarr_compression)
- {doc}`Agent Skills <agent_skills>`

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

about/install
about/intro
about/conventions
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
training
losses
finetuning
hooks
reporting
dynamics
```

```{toctree}
:caption: Distributed Simulations
:maxdepth: 1
:hidden:

distributed
distributed_shardtensor
distributed_byo
distributed_design
```

```{toctree}
:caption: Advanced Usage
:maxdepth: 1
:hidden:

distributed_training
zarr_compression
agent_skills
```
