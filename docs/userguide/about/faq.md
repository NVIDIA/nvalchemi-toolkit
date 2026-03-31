<!-- markdownlint-disable MD025 MD026 -->

# Frequently Asked Questions

## General

### How do I get started?

For installation instructions, check the [installation guide](install). For a
quick working example, see the [User Guide](../index). For detailed API usage,
refer to the [API documentation](../../modules/index).

If your question is not answered here, please submit a Github [Issue][issues_].

### What hardware does this support?

ALCHEMI Toolkit runs on:

- CUDA-capable NVIDIA GPUs (Compute Capability 8.0+, i.e. A100 and newer)
- CPU execution via NVIDIA Warp (x86 and ARM, including Apple Silicon)

For best performance, we recommend CUDA 12+ with driver version 570.xx or newer.
See the [installation guide](install) for full prerequisites.

## Dynamics & Simulation

### How do I compute neighbor lists during a simulation?

Register a {class}`~nvalchemi.dynamics.hooks.NeighborListHook` on your dynamics
engine. The hook recomputes neighbors at the `BEFORE_COMPUTE` stage of every
step (with optional Verlet-skin buffering to skip redundant rebuilds). See the
[dynamics hooks user guide](../dynamics_hooks) and the
{doc}`custom hook example </examples/advanced/02_custom_hook>` for a full
walkthrough.

### How do I run a distributed (multi-GPU) pipeline?

Chain dynamics stages with the `|` operator to build a
{class}`~nvalchemi.dynamics.DistributedPipeline`, then launch with `torchrun`.
See the [distributed pipeline API docs](../../modules/dynamics/distributed_pipeline)
and the
{doc}`distributed pipeline example </examples/distributed/01_distributed_pipeline>`
for a complete multi-rank setup.

## Models

### How do I download and use a MACE checkpoint?

Instantiate {py:class}`~nvalchemi.models.MACEPotential` with either a known
registry name (for example `"mace-mp-0b3-medium"`) or a local `.pt` path. The
registry-backed path handles download and cache resolution automatically. For
a full simulation example, see the
{doc}`MACE NVT example </examples/advanced/04_mace_nvt>` and the
[models user guide](../models).

## Interoperability

### How do I convert from ASE Atoms to nvalchemi?

Use {meth}`AtomicData.from_atoms() <nvalchemi.data.AtomicData.from_atoms>` to
convert an `ase.Atoms` object into an `AtomicData` graph. ASE integer tags are
automatically mapped to `AtomCategory` values. See the
{doc}`ASE integration example </examples/basic/03_ase_integration>` for a
round-trip workflow (ASE -> nvalchemi -> ASE).

[issues_]: https://www.github.com/NVIDIA/nvalchemi-toolkit/issues/new/choose
