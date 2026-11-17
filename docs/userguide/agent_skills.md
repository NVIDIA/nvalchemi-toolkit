<!-- markdownlint-disable MD014 -->

(agent_skills)=

# Agent Skills

The ALCHEMI Toolkit ships one **agent skill**, `nvalchemi-toolkit`. It helps
AI coding assistants (Claude Code, Codex, Cursor, and others) write and change
code that uses `nvalchemi`. The skill asks for missing scientific choices
before it runs anything, checks the installed version, and routes the agent
to a short note for the task at hand. Each note points to the user guide
pages and examples for the full details.

The skill lives in the repository under `skills/nvalchemi-toolkit/`:

| File | Covers |
|------|--------|
| `SKILL.md` | Scoping a request, the version check, routing, and reuse rules |
| `references/data.md` | {py:class}`~nvalchemi.data.AtomicData`, {py:class}`~nvalchemi.data.Batch`, Zarr storage, and data loading ({ref}`data_guide`, {ref}`datapipes_guide`) |
| `references/models.md` | Loading, wrapping, and combining models ({ref}`models_guide`) |
| `references/dynamics-optimization.md` | Relaxation, MD, hooks, reaction paths, and enhanced sampling ({ref}`dynamics_guide`) |
| `references/training.md` | Training, losses, fine-tuning, distillation, and multi-GPU training ({ref}`training_guide`) |
| `references/generative-workflows.md` | Generative sampling and generate-then-relax pipelines |
| `references/composition-and-extension.md` | Custom hooks, integrators, losses, biases, reporting, and domain decomposition |

## Installing the skill

**From a repository clone.** Copy or link the skill folder into the skills
directory of your agent, for example:

```bash
mkdir -p ~/.claude/skills
cp -r skills/nvalchemi-toolkit ~/.claude/skills/
```

Use your project's skills directory instead of the user-level one if you
prefer. The `evals/` subfolder holds the evaluation set and is not needed at
run time. Agents that read `AGENTS.md` are pointed to the skill from there, so
inside a clone they find it without an install.

**From the NVIDIA skills catalog.** Once the skill is listed in the
[NVIDIA skills catalog](https://github.com/NVIDIA/skills), you can install it
with the catalog's install command. Until then, use the clone install above.

The skill needs no credentials. It runs Python and shell commands to inspect
the installed package and to run small checks.
