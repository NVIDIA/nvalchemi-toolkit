<!-- markdownlint-disable MD014 -->

(agent_skills)=

# Agent Skills

The ALCHEMI Toolkit ships a set of **agent skills** --- concise instruction
files that AI coding assistants (Claude, Copilot, Cursor, etc.) can load to
get up to speed with the `nvalchemi` API without lengthy context-gathering.

Skills live in the repository under `.claude/skills/`.

## Installing skills

Point your AI assistant at the repository and ask it to install the skills
from `.claude/skills/`. Most assistants will auto-detect them once the
repository is open.

**Project-level install (recommended)** --- keeps the skills scoped to your
`nvalchemi` project so they are loaded only when you work in this repository.

**User-level install** --- if you work with `nvalchemi` frequently across
multiple checkouts or worktrees, you can install the skills in your user
configuration directory (e.g. `~/.claude/skills/`) so they are always
available.

## Available skills

| Skill | Description | Related user guide |
|-------|-------------|--------------------|
| `nvalchemi-data-structures` | How to use {py:class}`~nvalchemi.data.AtomicData` and {py:class}`~nvalchemi.data.Batch` for representing atomic systems and batching them for GPU computation. | {ref}`data_guide` |
| `nvalchemi-data-storage` | How to write, read, and load atomic data using the composable Zarr-backed storage pipeline (Writer, Reader, Dataset, DataLoader). | {ref}`datapipes_guide` |
| `nvalchemi-model-wrapping` | How to wrap an arbitrary MLIP using the {py:class}`~nvalchemi.models.base.BaseModelMixin` interface to standardize inputs, outputs, and embeddings. | {ref}`models_guide` |
| `nvalchemi-dynamics-api` | How to configure and run dynamics simulations, compose multi-stage pipelines ({py:class}`~nvalchemi.dynamics.FusedStage`, {py:class}`~nvalchemi.dynamics.DistributedPipeline`), use inflight batching, and manage data sinks. | {ref}`dynamics_guide` |
| `nvalchemi-dynamics-implementation` | How to implement a dynamics integrator by subclassing {py:class}`~nvalchemi.dynamics.base.BaseDynamics` and overriding `pre_update()` and `post_update()`. | {ref}`dynamics_guide` |
| `nvalchemi-dynamics-hooks` | How to use and write dynamics hooks --- callbacks that observe or modify batch state at specific points during each simulation step. | {ref}`hooks_guide` |
