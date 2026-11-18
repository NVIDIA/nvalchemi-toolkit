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

## Installing the skill (users)

Use this route if you write code with `nvalchemi` and do not need a clone of
the repository. It needs [Node.js](https://nodejs.org) for `npx`.

1. Install the toolkit itself, as described in the {ref}`install_guide`:

   ```bash
   pip install nvalchemi-toolkit
   ```

2. Install the skill from the
   [NVIDIA skills catalog](https://github.com/NVIDIA/skills):

   ```bash
   npx skills@latest add nvidia/skills --skill nvalchemi-toolkit
   ```

The skill and the toolkit are separate installs. The skill checks which
version of `nvalchemi` is installed before it gives advice.

To choose a coding agent or an installation scope, select them below to
generate the command.

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
      <pre><code id="skills-install-command">npx skills@latest add nvidia/skills --skill nvalchemi-toolkit</code></pre>
      <button type="button" id="copy-skills-command" aria-label="Copy install command">Copy</button>
    </div>
  </div>
  <p class="install-matrix-note" id="skills-matrix-note" aria-live="polite"></p>
</div>
<!-- markdownlint-enable MD013 -->

<noscript>
Install the skill from the NVIDIA catalog with:
<code>npx skills@latest add nvidia/skills --skill nvalchemi-toolkit</code>
</noscript>

## Coding Agent Instructions

Different coding assistants load skills from different folders.
Without `--global`, the skill installs into the current project. Run the
command from your project root.

### Codex

Codex loads skills from `.agents/skills/` in the project root or
`~/.agents/skills/` in the user home directory.
To install the skill for Codex in the current project, run:

```bash
npx skills@latest add nvidia/skills --skill nvalchemi-toolkit --agent codex
```

To install the skill for all projects, add the `--global` flag:

```bash
npx skills@latest add nvidia/skills --skill nvalchemi-toolkit --agent codex --global
```

### Claude Code

Claude Code loads skills from `.claude/skills/` in the project root or
`~/.claude/skills/` in the user home directory.
To install the skill for Claude Code in the current project, run:

```bash
npx skills@latest add nvidia/skills --skill nvalchemi-toolkit --agent claude-code
```

To install the skill for all projects, add the `--global` flag:

```bash
npx skills@latest add nvidia/skills --skill nvalchemi-toolkit --agent claude-code --global
```

If you install the skill during an active Claude Code session, run
`/reload-skills` to load the skill.
Use Claude Code `/skills` to confirm the skill is listed.

### Pi

Pi loads skills from `.agents/skills/` in the project root or
`~/.agents/skills/` in the user home directory, the same folders as Codex.
To install the skill for Pi in the current project, run:

```bash
npx skills@latest add nvidia/skills --skill nvalchemi-toolkit --agent pi
```

To install the skill for all projects, add the `--global` flag:

```bash
npx skills@latest add nvidia/skills --skill nvalchemi-toolkit --agent pi --global
```

### Updating

Update the skill and the toolkit separately:

```bash
npx skills update            # refresh installed skills
npx skills check             # preview what is out of date first
pip install -U nvalchemi-toolkit
```

Copies placed by hand do not update with `npx skills update`.

## Working on the skill (developers)

Use this route if you cloned the repository to change the skill or the
toolkit. The skill lives in `skills/nvalchemi-toolkit/`. Edit it there. To try
your edits in your coding agent, copy the folder into the agent's skills
directory:

```bash
# Claude Code
mkdir -p ~/.claude/skills
cp -r skills/nvalchemi-toolkit ~/.claude/skills/

# Codex and Pi
mkdir -p ~/.agents/skills
cp -r skills/nvalchemi-toolkit ~/.agents/skills/
```

Copy again after each edit. The `evals/` folder holds the evaluation set and is
not needed at run time. The skill needs no credentials. It runs Python and
shell commands to inspect the installed package and to run small tests.

Contributors also use `AGENTS.md` at the repository root, which holds the setup
steps and conventions for working in the repository. Claude Code reads
`CLAUDE.md` instead, which imports `@AGENTS.md`. See "Maintaining the skill" in
`AGENTS.md` for the rules on editing the skill.
