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
"""Tests for the ``nvalchemi-skills`` CLI (:mod:`nvalchemi.skills`)."""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from click.testing import CliRunner

from nvalchemi import skills as skills_mod

REPO_SKILLS = Path(__file__).resolve().parents[2] / ".claude" / "skills"
SKILL_NAMES = sorted(p.name for p in REPO_SKILLS.iterdir() if p.is_dir())


class TestSkillSourceResolution:
    """Resolution of the bundled-skills directory."""

    def test_dev_checkout_fallback(self):
        """In an editable checkout the repo .claude/skills is used."""
        root = skills_mod._bundled_skills_root()
        assert root == REPO_SKILLS

    def test_bundled_dir_preferred(self, tmp_path, monkeypatch):
        """A wheel-style nvalchemi/_skills directory wins over the fallback."""
        fake_pkg = tmp_path / "nvalchemi"
        bundled = fake_pkg / "_skills" / "nvalchemi-demo"
        bundled.mkdir(parents=True)
        (bundled / "SKILL.md").write_text("---\nname: nvalchemi-demo\n---\n")
        monkeypatch.setattr(
            skills_mod, "__file__", str(fake_pkg / "skills.py"), raising=False
        )
        assert skills_mod._bundled_skills_root() == fake_pkg / "_skills"

    def test_missing_everything_raises(self, tmp_path, monkeypatch):
        """A clear error is raised when no skills can be located."""
        monkeypatch.setattr(
            skills_mod,
            "__file__",
            str(tmp_path / "nvalchemi" / "skills.py"),
            raising=False,
        )
        with pytest.raises(Exception, match="No bundled skills found"):
            skills_mod._bundled_skills_root()


class TestListCommand:
    """``nvalchemi-skills list``."""

    def test_lists_all_skills(self):
        """Every repo skill appears in the listing with a summary."""
        result = CliRunner().invoke(skills_mod.main, ["list"])
        assert result.exit_code == 0
        for name in SKILL_NAMES:
            assert name in result.output


class TestInstallCommand:
    """``nvalchemi-skills install``."""

    def test_install_dest_copies_everything(self, tmp_path):
        """--dest copies each skill directory content-identically."""
        result = CliRunner().invoke(
            skills_mod.main, ["install", "--dest", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
        for name in SKILL_NAMES:
            comparison = filecmp.dircmp(REPO_SKILLS / name, tmp_path / name)
            assert not comparison.left_only
            assert not comparison.right_only
            assert not comparison.diff_files

    def test_rerun_without_force_skips(self, tmp_path):
        """Existing skills are preserved unless --force is passed."""
        runner = CliRunner()
        runner.invoke(skills_mod.main, ["install", "--dest", str(tmp_path)])
        marker = tmp_path / SKILL_NAMES[0] / "SKILL.md"
        marker.write_text("locally modified")
        result = runner.invoke(skills_mod.main, ["install", "--dest", str(tmp_path)])
        assert result.exit_code == 0
        assert "use --force to replace" in result.output
        assert marker.read_text() == "locally modified"

    def test_force_overwrites(self, tmp_path):
        """--force replaces locally modified copies."""
        runner = CliRunner()
        runner.invoke(skills_mod.main, ["install", "--dest", str(tmp_path)])
        marker = tmp_path / SKILL_NAMES[0] / "SKILL.md"
        marker.write_text("locally modified")
        result = runner.invoke(
            skills_mod.main, ["install", "--dest", str(tmp_path), "--force"]
        )
        assert result.exit_code == 0
        assert marker.read_text() != "locally modified"

    def test_project_scope_resolves_cwd(self, tmp_path, monkeypatch):
        """--target claude --scope project installs to <cwd>/.claude/skills."""
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            skills_mod.main, ["install", "--target", "claude", "--scope", "project"]
        )
        assert result.exit_code == 0, result.output
        for name in SKILL_NAMES:
            assert (tmp_path / ".claude" / "skills" / name / "SKILL.md").is_file()

    def test_user_scope_unsupported_target_errors(self):
        """copilot has no user-scope directory and must error clearly."""
        result = CliRunner().invoke(
            skills_mod.main, ["install", "--target", "copilot", "--scope", "user"]
        )
        assert result.exit_code != 0
        assert "no user-scope skills directory" in result.output

    def test_auto_detects_existing_agent_dirs(self, tmp_path, monkeypatch):
        """--target auto installs into agent dirs present in the project."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".cursor").mkdir()
        monkeypatch.setattr(
            skills_mod,
            "TARGET_DIRS",
            {"cursor": (".cursor/skills", None)},
        )
        result = CliRunner().invoke(skills_mod.main, ["install"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / ".cursor" / "skills" / SKILL_NAMES[0]).is_dir()

    def test_self_install_does_not_clobber_source(self, monkeypatch):
        """Installing into the repo's own .claude/skills is a no-op."""
        monkeypatch.chdir(REPO_SKILLS.parents[1])
        result = CliRunner().invoke(
            skills_mod.main,
            ["install", "--target", "claude", "--scope", "project", "--force"],
        )
        assert result.exit_code == 0, result.output
        assert "source and destination match" in result.output
