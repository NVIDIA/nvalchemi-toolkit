(() => {
  const matrix = document.getElementById("skills-matrix");
  if (!matrix) return;

  const command = matrix.querySelector("#skills-install-command");
  const copyButton = matrix.querySelector("#copy-skills-command");
  const note = matrix.querySelector("#skills-matrix-note");

  const notes = {
    auto: "Auto-detects active coding agents in your project.",
    codex: "Installs into the Codex skills directory (.agents/skills/ or ~/.agents/skills/).",
    "claude-code": "Installs into the Claude Code skills directory (.claude/skills/ or ~/.claude/skills/). In Claude Code, run /reload-skills to refresh.",
    pi: "Installs into the Pi skills directory (.agents/skills/ or ~/.agents/skills/).",
  };

  function selected(name) {
    const input = matrix.querySelector(`input[name="${name}"]:checked`);
    return input ? input.value : "";
  }

  function updateCommand() {
    const agent = selected("agent");
    const scope = selected("scope");

    const parts = ["npx", "skills@latest", "add", "nvidia/skills", "--skill", "nvalchemi-toolkit"];

    if (agent && agent !== "auto") {
      parts.push("--agent", agent);
    }

    if (scope === "global") {
      parts.push("--global");
    }

    command.textContent = parts.join(" ");
    if (note && notes[agent]) {
      note.textContent = notes[agent];
    }
  }

  matrix.addEventListener("change", updateCommand);
  if (copyButton) {
    copyButton.addEventListener("click", async () => {
      await navigator.clipboard.writeText(command.textContent);
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = "Copy";
      }, 1500);
    });
  }

  updateCommand();
})();
