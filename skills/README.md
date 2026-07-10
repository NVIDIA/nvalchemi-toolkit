# Agent Skills for `nvalchemi`

This folder contains task-focused guides ("skills") for working with the
`nvalchemi` API. Each skill is a plain Markdown file with YAML frontmatter,
readable by any coding agent or human. Agents that read `.claude/skills` or
`.agents/skills` discover them automatically inside a clone (both are
symlinks to `skills/`); other agents reach them through the routing table in
the repository's `AGENTS.md`, and installed-package users can copy them
locally (see "Installing the skills" below).

## Which skill do I need?

| Task | Skill |
| --- | --- |
| Build or batch atomic systems; debug shape, dtype, or device errors | `nvalchemi-data-structures` |
| Write, read, compose, or stream atomic data with the Zarr pipeline | `nvalchemi-data-storage` |
| Tune Dataset/DataLoader throughput or Zarr chunking | `nvalchemi-zarr-perf` |
| Wrap an MLIP or custom PyTorch model for use in nvalchemi | `nvalchemi-model-wrapping` |
| Train a model from scratch (strategy, optimizers, validation) | `nvalchemi-training-api` |
| Adapt a pretrained model to new reference data | `nvalchemi-fine-tuning` |
| Choose, weight, mask, or implement loss functions | `nvalchemi-loss-api` |
| Scale training across GPUs or nodes (DDP, rank safety) | `nvalchemi-distributed-training` |
| Run MD, relaxation, or EOS scans; compose batched pipelines | `nvalchemi-dynamics-api` |
| Add per-step callbacks (neighbor lists, convergence, logging) | `nvalchemi-dynamics-hooks` |
| Implement a new integrator, optimizer, or sampler class | `nvalchemi-dynamics-implementation` |
| Add progress dashboards, TensorBoard, or CSV observability | `nvalchemi-reporting` |

Each skill lives at `skills/<name>/SKILL.md` (the canonical location). For
in-repo discovery, `.claude/skills` and `.agents/skills` are symlinks to
`skills/`, so agents that read those paths (Claude Code, Cursor, OpenCode,
Gemini CLI, and any agentskills.io tool) find them with no setup.

## How the skills relate

Skills build on each other in a few short chains; read upstream skills first
when you are new to an area:

```text
data-structures -> data-storage -> zarr-perf
model-wrapping + loss-api -> training-api -> fine-tuning
training-api -> distributed-training
dynamics-hooks -> dynamics-api | dynamics-implementation
reporting (orthogonal: attaches to training and dynamics)
```

## Installing the skills

- **Inside a clone**: nothing to do for agents that read `.claude/skills` or
  `.agents/skills` (Claude Code, Cursor, OpenCode, Gemini CLI) — the committed
  symlinks point them at `skills/`. For an agent that only reads its own
  directory (for example Codex `.codex/skills` or Copilot `.github/skills`),
  run `nvalchemi-skills install --target <agent>`.
- **From an installed package**: the skills ship inside the `nvalchemi`
  wheel. Run `nvalchemi-skills install` to copy them into your project's or
  home agent directories (`-h`/`--help` lists targets). Installed skills match
  the API version of the package you installed, and a `.nvalchemi-skills.json`
  manifest is written alongside them recording that version and its provenance
  (a released `wheel`, or a `repository` checkout with its `git describe`).
- **Straight from GitHub** (any agent, no Python needed):
  `npx skills add NVIDIA/nvalchemi-toolkit`.

## Authoring conventions

Follow these rules when adding or editing a skill.

### Frontmatter

Exactly two YAML fields — no agent-specific fields (such as
`allowed-tools`), so the skills stay portable across agent platforms:

- `name`: must equal the directory name; lowercase words joined by
  hyphens, prefixed `nvalchemi-`.
- `description`: a `>-` folded scalar, at most ~500 characters. The first
  sentence states what the skill teaches; it must also contain a
  "Use when ..." clause listing concrete task triggers, so agents can route
  without opening the file.

### Structure

- One H1 title after the frontmatter.
- `## Overview` comes first: what the skill covers, why, and pointers to
  the deeper `docs/userguide/` pages.
- Then task-oriented H2 sections in workflow order.
- `## Troubleshooting` (optional but recommended for complex areas) goes
  second-to-last, formatted as a `| Symptom | Cause | Fix |` table with
  error messages quoted verbatim from the source.
- `## Key files` (optional) goes last.

### Content rules

- Keep the whole skill in one `SKILL.md` (200–550 lines). No side files:
  single-file skills stay portable and are read in full by every agent.
- Code examples must be runnable against the current API: every import,
  class, function, and argument must exist in the source tree. Verify
  before committing.
- Every repository path mentioned must exist.
- Wrap prose and code at 88 characters (markdownlint MD013 applies to code
  blocks; tables are exempt).
- Address the reader directly ("you", imperative). Never write "agents
  should" or "load the skill" — write "see the `<name>` skill".
- Cross-reference sibling skills by backticked name; reference repository
  files by their path from the repo root.

### When adding a skill

Update all three indexes:

1. The routing table in this README.
2. The skill routing table in `AGENTS.md`.
3. The table in `docs/userguide/agent_skills.md`.

Then run `uv run pre-commit run markdownlint -a` (note: `make lint` does
not run markdownlint) and verify the frontmatter parses as YAML.
