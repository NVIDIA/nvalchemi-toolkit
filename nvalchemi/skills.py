# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install the agent skills bundled with the ``nvalchemi`` package.

The task-focused agent skills that live in ``.claude/skills/`` of the
``nvalchemi-toolkit`` repository ship inside the ``nvalchemi`` wheel. This
module provides the ``nvalchemi-skills`` console command to list them and to
copy them into the skill directories used by coding agents (Claude Code,
Cursor, Codex, GitHub Copilot, OpenCode, Gemini CLI, or the tool-agnostic
``.agents`` layout), so the guidance matches the installed API version.

Run with::

    nvalchemi-skills list
    nvalchemi-skills install                  # auto-detect agents in cwd/home
    nvalchemi-skills install --target claude --scope user
    nvalchemi-skills install --dest /path/to/skills --force
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click

#: Per-agent skill directories: target name -> (project-scope, user-scope).
#: A ``None`` user scope means the agent has no user-level skills directory.
TARGET_DIRS: dict[str, tuple[str, str | None]] = {
    "claude": (".claude/skills", "~/.claude/skills"),
    "cursor": (".cursor/skills", "~/.cursor/skills"),
    "codex": (".codex/skills", "~/.codex/skills"),
    "copilot": (".github/skills", None),
    "opencode": (".opencode/skill", "~/.config/opencode/skill"),
    "gemini": (".gemini/skills", "~/.gemini/skills"),
    "agents": (".agents/skills", "~/.agents/skills"),
}


def _bundled_skills_root() -> Path:
    """Locate the skills shipped with this installation.

    Returns
    -------
    Path
        Directory containing one subdirectory per skill.

    Raises
    ------
    click.ClickException
        If neither the wheel-bundled ``nvalchemi/_skills`` directory nor the
        repository ``.claude/skills`` fallback (editable installs) exists.
    """
    pkg_dir = Path(__file__).resolve().parent
    bundled = pkg_dir / "_skills"
    if bundled.is_dir():
        return bundled
    repo_skills = pkg_dir.parent / ".claude" / "skills"
    if repo_skills.is_dir():
        return repo_skills
    raise click.ClickException(
        "No bundled skills found. Expected them at "
        f"{bundled} (wheel install) or {repo_skills} (repository checkout)."
    )


def _skill_dirs(root: Path) -> list[Path]:
    """List the individual skill directories under ``root``.

    Parameters
    ----------
    root : Path
        Directory returned by :func:`_bundled_skills_root`.

    Returns
    -------
    list of Path
        Sorted skill directories (each contains a ``SKILL.md``).
    """
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "SKILL.md").is_file()
    )


def _frontmatter_summary(skill_dir: Path) -> str:
    """Extract the first sentence of a skill's frontmatter description.

    Parameters
    ----------
    skill_dir : Path
        Directory containing a ``SKILL.md`` with YAML frontmatter.

    Returns
    -------
    str
        First sentence of the ``description`` field, or an empty string if
        the frontmatter cannot be parsed.
    """
    lines = (skill_dir / "SKILL.md").read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    collected: list[str] = []
    in_description = False
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("description:"):
            in_description = True
            continue
        if in_description:
            if line.startswith((" ", "\t")):
                collected.append(line.strip())
            else:
                break
    text = " ".join(collected)
    sentence, _, _ = text.partition(". ")
    return sentence + "." if sentence and not sentence.endswith(".") else sentence


def _detect_targets(project_root: Path) -> list[str]:
    """Auto-detect agents by looking for their config directories.

    Parameters
    ----------
    project_root : Path
        Directory treated as the project root (usually the cwd).

    Returns
    -------
    list of str
        Target names whose parent config directory (for example ``.cursor``)
        exists in ``project_root`` or in the home directory. Defaults to
        ``["claude"]`` when nothing is detected.
    """
    detected = []
    for target, (project_rel, user_dir) in TARGET_DIRS.items():
        parent = Path(project_rel).parts[0]
        if (project_root / parent).is_dir():
            detected.append(target)
        elif user_dir is not None and Path(user_dir).expanduser().parent.is_dir():
            detected.append(target)
    return detected or ["claude"]


def _resolve_destination(target: str, scope: str) -> Path:
    """Map a target/scope pair to a concrete skills directory.

    Parameters
    ----------
    target : str
        A key of :data:`TARGET_DIRS`.
    scope : str
        Either ``"project"`` (relative to the cwd) or ``"user"``.

    Returns
    -------
    Path
        Destination directory for the skills.

    Raises
    ------
    click.ClickException
        If the target has no user-scope directory and ``scope="user"``.
    """
    project_rel, user_dir = TARGET_DIRS[target]
    if scope == "project":
        return Path.cwd() / project_rel
    if user_dir is None:
        raise click.ClickException(
            f"Target '{target}' has no user-scope skills directory; "
            "use --scope project or --dest."
        )
    return Path(user_dir).expanduser()


def _install_into(source_root: Path, dest: Path, force: bool) -> tuple[int, int]:
    """Copy every bundled skill directory into ``dest``.

    Parameters
    ----------
    source_root : Path
        Directory containing the bundled skills.
    dest : Path
        Destination skills directory (created if missing).
    force : bool
        Replace skill directories that already exist at the destination.

    Returns
    -------
    tuple of int
        ``(copied, skipped)`` counts.
    """
    dest = dest.expanduser().resolve()
    copied = skipped = 0
    for skill_dir in _skill_dirs(source_root):
        if skill_dir.resolve() == (dest / skill_dir.name).resolve():
            skipped += 1
            click.echo(f"  = {skill_dir.name} (source and destination match)")
            continue
        target_dir = dest / skill_dir.name
        if target_dir.exists():
            if not force:
                skipped += 1
                click.echo(f"  - {skill_dir.name} (exists, use --force to replace)")
                continue
            shutil.rmtree(target_dir)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(skill_dir, target_dir)
        copied += 1
        click.echo(f"  + {skill_dir.name} -> {target_dir}")
    return copied, skipped


@click.group()
def main() -> None:
    """Manage the agent skills bundled with the nvalchemi package."""


@main.command("list")
def list_skills() -> None:
    """List the bundled skills and their one-line summaries."""
    root = _bundled_skills_root()
    skills = _skill_dirs(root)
    click.echo(f"{len(skills)} skills bundled ({root}):")
    for skill_dir in skills:
        click.echo(f"  {skill_dir.name}: {_frontmatter_summary(skill_dir)}")


@main.command()
@click.option(
    "--target",
    type=click.Choice([*TARGET_DIRS, "all", "auto"]),
    default="auto",
    show_default=True,
    help="Agent whose skills directory receives the copies.",
)
@click.option(
    "--scope",
    type=click.Choice(["project", "user"]),
    default="project",
    show_default=True,
    help="Install relative to the current project or the home directory.",
)
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit destination directory; overrides --target/--scope.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace skills that already exist at the destination.",
)
def install(target: str, scope: str, dest: Path | None, force: bool) -> None:
    """Copy the bundled skills into agent skill directories.

    With the default ``--target auto`` the command detects which agent
    directories already exist (in the current project or the home directory)
    and installs to each of them; it falls back to Claude Code's layout when
    none are found.
    """
    root = _bundled_skills_root()
    if dest is not None:
        destinations = [dest]
    else:
        if target == "auto":
            targets = _detect_targets(Path.cwd())
        elif target == "all":
            targets = list(TARGET_DIRS)
        else:
            targets = [target]
        destinations = []
        for name in targets:
            try:
                destinations.append(_resolve_destination(name, scope))
            except click.ClickException:
                if target != name:  # implicit via auto/all: skip quietly
                    continue
                raise
    total_copied = total_skipped = 0
    for destination in destinations:
        click.echo(f"Installing into {destination}:")
        copied, skipped = _install_into(root, destination, force)
        total_copied += copied
        total_skipped += skipped
    click.echo(f"Done: {total_copied} copied, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
