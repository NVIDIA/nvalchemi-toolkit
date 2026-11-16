<!-- markdownlint-disable MD025 MD033 MD014 -->

(agent_skills)=

# Agent Skills

The ALCHEMI Toolkit ships one agent skill, `nvalchemi-toolkit`.
An agent skill is a directory of instructions for AI coding assistants.
The skill teaches assistants (such as Codex, Claude Code, and Pi) how to write
and modify code that uses `nvalchemi`.
The skill asks for missing scientific parameters before it runs code, inspects
the installed version, and routes the assistant to reference notes.
Each note links to the user guide pages and examples for complete details.

The skill contains the following reference files:

<!-- markdownlint-disable MD013 -->
| File | Covers |
|------|--------|
| `SKILL.md` | Scoping a request, version inspection, routing, and reuse rules |
| `references/data.md` | {py:class}`~nvalchemi.data.AtomicData`, {py:class}`~nvalchemi.data.Batch`, Zarr storage, and data loading ({ref}`data_guide`, {ref}`datapipes_guide`) |
| `references/models.md` | Loading, wrapping, and combining models ({ref}`models_guide`) |
| `references/dynamics-optimization.md` | Relaxation, MD, hooks, reaction paths, and biased sampling ({ref}`dynamics_guide`) |
| `references/training.md` | Training, losses, fine-tuning, distillation, and multi-GPU training ({ref}`training_guide`) |
| `references/generative-workflows.md` | Generative sampling and generate-then-relax pipelines |
| `references/composition-and-extension.md` | Custom hooks, integrators, losses, biases, reporting, and domain decomposition |
<!-- markdownlint-enable MD013 -->

## Installing from the NVIDIA Skills Catalog

You can install the skill from the
[NVIDIA skills catalog](https://github.com/NVIDIA/skills) with `npx`.
Select your coding agent and installation scope below to generate the install
command.

<!-- markdownlint-disable MD013 -->
<div class="install-matrix" id="skills-matrix">
  <div class="install-matrix-row">
    <div class="install-matrix-heading" id="agent-label">Coding agent</div>
    <div class="install-matrix-options" role="radiogroup" aria-labelledby="agent-label">
      <input class="install-matrix-input" type="radio" name="agent" id="agent-auto" value="auto" checked>
      <label for="agent-auto">Auto-detect</label>
      <input class="install-matrix-input" type="radio" name="agent" id="agent-codex" value="codex">
      <label for="agent-codex">Codex</label>
      <input class="install-matrix-input" type="radio" name="agent" id="agent-claude" value="claude-code">
      <label for="agent-claude">Claude Code</label>
      <input class="install-matrix-input" type="radio" name="agent" id="agent-pi" value="pi">
      <label for="agent-pi">Pi</label>
    </div>
  </div>
  <div class="install-matrix-row">
    <div class="install-matrix-heading" id="scope-label">Scope</div>
    <div class="install-matrix-options" role="radiogroup" aria-labelledby="scope-label">
      <input class="install-matrix-input" type="radio" name="scope" id="scope-project" value="project" checked>
      <label for="scope-project">Project</label>
      <input class="install-matrix-input" type="radio" name="scope" id="scope-global" value="global">
      <label for="scope-global">Global</label>
    </div>
  </div>
  <div class="install-matrix-output">
    <div class="install-matrix-heading">Run this command</div>
    <div class="install-command">
      <pre><code id="skills-install-command">npx skills add nvidia/skills --skill nvalchemi-toolkit</code></pre>
      <button type="button" id="copy-skills-command" aria-label="Copy install command">Copy</button>
    </div>
  </div>
  <p class="install-matrix-note" id="skills-matrix-note" aria-live="polite"></p>
</div>
<!-- markdownlint-enable MD013 -->

<noscript>
Install the skill from the NVIDIA catalog with:
<code>npx skills add nvidia/skills --skill nvalchemi-toolkit</code>
</noscript>

## Coding Agent Instructions

Different coding assistants store skills and repository instructions in
different locations.

### Codex

Codex loads skills from `.agents/skills/` in the project root or
`~/.agents/skills/` in the user home directory.
To install the skill for Codex in the current project, run:

```bash
npx skills add nvidia/skills --skill nvalchemi-toolkit --agent codex
```

To install the skill for all projects, add the `--global` flag:

```bash
npx skills add nvidia/skills --skill nvalchemi-toolkit --agent codex --global
```

Codex reads repository conventions from `AGENTS.md` automatically.

### Claude Code

Claude Code loads skills from `.claude/skills/` in the project root or
`~/.claude/skills/` in the user home directory.
To install the skill for Claude Code in the current project, run:

```bash
npx skills add nvidia/skills --skill nvalchemi-toolkit --agent claude-code
```

To install the skill for all projects, add the `--global` flag:

```bash
npx skills add nvidia/skills --skill nvalchemi-toolkit --agent claude-code --global
```

If you install the skill during an active Claude Code session, run
`/reload-skills` to load the skill.
Claude Code reads repository guidance from `CLAUDE.md`.
In this repository, `CLAUDE.md` imports `@AGENTS.md`.

### Pi

Pi loads skills from `.pi/agent/skills/` in the project root or
`~/.pi/agent/skills/` in the user home directory.
To install the skill for Pi in the current project, run:

```bash
npx skills add nvidia/skills --skill nvalchemi-toolkit --agent pi
```

To install the skill for all projects, add the `--global` flag:

```bash
npx skills add nvidia/skills --skill nvalchemi-toolkit --agent pi --global
```

Pi reads repository guidance from `AGENTS.md` automatically.

## Installing from a Clone

If you develop the skill locally, copy or link the skill folder into the skills
directory of your agent.

```bash
mkdir -p ~/.agents/skills
cp -r skills/nvalchemi-toolkit ~/.agents/skills/
```

The `evals/` folder contains benchmark tests for the skill and is not needed at
run time.
The skill needs no credentials.
It operates Python and shell commands to inspect the installed package and to
run small tests.
