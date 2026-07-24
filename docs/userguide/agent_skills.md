<!-- markdownlint-disable MD014 -->

(agent_skills)=

# Agent Skills

The ALCHEMI Toolkit ships a set of **agent skills** --- concise instruction
files that AI coding assistants (Claude, Copilot, Cursor, etc.) can load to
get up to speed with the `nvalchemi` API without lengthy context-gathering.

Skills live in the repository under `.claude/skills/`.

## Installing skills

**Inside a repository clone** --- nothing to install. Claude Code discovers
`.claude/skills/` automatically; other agents are routed to the right
`SKILL.md` by the table in the repository's `AGENTS.md`.

**Outside a clone** --- copy the skill folders from `.claude/skills/` into
your project's skills directory, or your user-level one (e.g.
`~/.claude/skills/`) if you work with `nvalchemi` across many checkouts.

## Available skills

| Skill | Description | Related user guide |
|-------|-------------|--------------------|
| `nvalchemi-data-structures` | How to use {py:class}`~nvalchemi.data.AtomicData` and {py:class}`~nvalchemi.data.Batch` for representing atomic systems and batching them for GPU computation. | {ref}`data_guide` |
| `nvalchemi-data-storage` | How to write, read, compose, and load atomic data using the composable Zarr-backed storage pipeline (Writer, Reader, Dataset, MultiDataset, DataLoader). | {ref}`datapipes_guide` |
| `nvalchemi-zarr-perf` | How to tune Zarr-backed Reader, Dataset, MultiDataset, and DataLoader throughput with fused reads, validation skipping, pinned memory, and benchmark sweeps. | {ref}`read_performance_tuning` |
| `nvalchemi-model-wrapping` | How to wrap an arbitrary MLIP using the {py:class}`~nvalchemi.models.base.BaseModelMixin` interface to standardize inputs, outputs, and embeddings. | {ref}`models_guide` |
| `nvalchemi-training-api` | How to configure, run, extend, and debug {py:class}`~nvalchemi.training.TrainingStrategy` workflows. | {ref}`training_guide` |
| `nvalchemi-loss-api` | How to use built-in loss terms, compose training objectives, and implement custom loss functions. | {ref}`losses_guide` |
| `nvalchemi-fine-tuning` | How to configure fine-tuning workflows and adapt pretrained checkpoints through the CLI or API. | {ref}`finetuning_guide` |
| `nvalchemi-dynamics-api` | How to configure and run dynamics simulations, compose multi-stage pipelines ({py:class}`~nvalchemi.dynamics.FusedStage`, {py:class}`~nvalchemi.dynamics.DistributedPipeline`), use inflight batching, and manage data sinks. | {ref}`dynamics_guide` |
| `nvalchemi-dynamics-implementation` | How to implement a dynamics integrator by subclassing {py:class}`~nvalchemi.dynamics.base.BaseDynamics` and overriding `pre_update()` and `post_update()`. | {ref}`dynamics_guide` |
| `nvalchemi-dynamics-hooks` | How to use and write dynamics hooks --- callbacks that observe or modify batch state at specific points during each simulation step. | {ref}`hooks_guide` |
| `nvalchemi-training-api` | How to configure training workflows with {py:class}`~nvalchemi.training.TrainingStrategy`: losses, optimizers, schedulers, validation, hooks, restartable checkpoints, and scaling to multiple GPUs or nodes with `DistributedManager` and `DDPHook`. | {ref}`training_guide` |
| `nvalchemi-fine-tuning` | How to adapt a pretrained model to new reference data with {py:class}`~nvalchemi.training.FineTuningStrategy`, module patches, and trainable-parameter filters. | {ref}`finetuning_guide` |
| `nvalchemi-loss-api` | How to use built-in loss functions and implement custom losses --- residual types, per-atom normalization, masking, and graph-balanced reductions. | {ref}`losses_guide` |
| `nvalchemi-reporting` | How to add observability with `ReportingOrchestrator`, `RichReporter`, `TensorBoardReporter`, and the dynamics `LoggingHook`. | {ref}`reporting_guide` |
