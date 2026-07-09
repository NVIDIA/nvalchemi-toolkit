(() => {
  const matrix = document.getElementById("install-matrix");
  if (!matrix) return;

  const command = matrix.querySelector("#install-command");
  const copyButton = matrix.querySelector("#copy-install-command");
  const note = matrix.querySelector("#install-matrix-note");
  const extraInputs = [...matrix.querySelectorAll('input[name="extra"]')];
  const maceInput = matrix.querySelector("#extra-mace");
  const umaInput = matrix.querySelector("#extra-uma");
  const cudaInputs = [
    matrix.querySelector("#accelerator-cu12"),
    matrix.querySelector("#accelerator-cu13"),
  ];
  const torchBackends = { none: "cpu", cu12: "cu126", cu13: "cu130" };
  const pipTorchIndexes = {
    cu12: "https://download.pytorch.org/whl/cu126",
    cu13: "https://download.pytorch.org/whl/cu130",
  };

  function selected(name) {
    return matrix.querySelector(`input[name="${name}"]:checked`).value;
  }

  function selectedExtras() {
    return extraInputs.filter((input) => input.checked).map((input) => input.value);
  }

  function updateConstraints(accelerator, extras) {
    const hasMace = extras.includes("mace");
    const hasUma = extras.includes("uma");

    cudaInputs.forEach((input) => {
      input.disabled = hasUma;
    });
    maceInput.disabled = hasUma;
    umaInput.disabled = accelerator !== "none" || hasMace;

    if (hasUma) {
      note.textContent = "UMA uses a separate environment and cannot be combined with CUDA or MACE.";
    } else if (umaInput.disabled) {
      note.textContent = "Clear CUDA and MACE to make UMA available.";
    } else {
      note.textContent = "Choose any combination of compatible optional extras.";
    }
  }

  function updateCommand() {
    const packageManager = selected("package-manager");
    const accelerator = selected("accelerator");
    const extras = selectedExtras();
    const allExtras = accelerator === "none" ? extras : [accelerator, ...extras];
    const packageSpec = allExtras.length
      ? `nvalchemi-toolkit[${allExtras.join(",")}]`
      : "nvalchemi-toolkit";

    if (packageManager === "uv") {
      command.textContent = `uv pip install --torch-backend ${torchBackends[accelerator]} '${packageSpec}'`;
    } else {
      const torchIndex = pipTorchIndexes[accelerator];
      command.textContent = torchIndex
        ? `pip install --extra-index-url ${torchIndex} '${packageSpec}'`
        : `pip install '${packageSpec}'`;
    }
    updateConstraints(accelerator, extras);
  }

  matrix.addEventListener("change", updateCommand);
  copyButton.addEventListener("click", async () => {
    await navigator.clipboard.writeText(command.textContent);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1500);
  });

  updateCommand();
})();
