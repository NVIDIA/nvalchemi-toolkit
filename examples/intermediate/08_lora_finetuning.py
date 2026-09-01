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
"""
LoRA Fine-Tuning MACE on LPSC dataset
=====================================

This example illustrates how to fine-tune a model with LoRA adapters using
:class:`~nvalchemi.training.FineTuningStrategy` with
:class:`~nvalchemi.training.peft.lora.LoRAConfig`. It fine-tunes the MACE ``medium-mpa-0``
foundation model on the ``li/data/LPSC_600.extxyz`` subset from the public
``ev-tlt/MACE_finetuning_supplementary`` Hugging Face dataset.
Dataset attribution: ``li/data/LPSC_600.extxyz`` comes from
``ev-tlt/MACE_finetuning_supplementary``; the dataset lists ``li/data/`` as
CC BY 4.0, sourced from Zenodo 15686940 / Kim et al. See
https://huggingface.co/datasets/ev-tlt/MACE_finetuning_supplementary and
https://creativecommons.org/licenses/by/4.0/.

The workflow is as follows:

#. Prepare the LPSC_600 dataset by downloading ``LPSC_600.extxyz`` and
   converting the structures into :class:`~nvalchemi.data.AtomicData` objects.
#. Split the data into training and validation subsets.
#. Prepare the MACE ``medium-mpa-0`` foundation model for fine-tuning by
   fitting residual atomic reference-energy corrections and updating the model
   with the refitted values.
#. Inspect the registered LoRA wrappers to see which layer types can receive
   adapters.
#. Construct the LoRA fine-tuning strategy, attach adapters, and train with
   validation reporting.
#. Save a trainable-only PEFT checkpoint, reload it into a fresh foundation
   model, and run validation inference.

It can be run on one GPU with the following command:

.. code-block:: bash

   uv run --extra cu12 --extra ase --extra mace \
       --with huggingface-hub \
       python examples/intermediate/08_lora_finetuning.py

Or use ``--extra cu13`` to match your CUDA environment. The runtime constants
below keep the example small. Increase ``TRAINING_EPOCHS`` for a longer run.

"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import iread
from huggingface_hub import hf_hub_download

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import (
    DataLoader,
    InMemoryDataset,
)
from nvalchemi.hooks import (
    HookContext,
    NeighborListHook,
    ReportingOrchestrator,
    RichReporter,
)
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.training import (
    ComposedLossFunction,
    EnergyHuberLoss,
    FineTuningStrategy,
    ForceHuberLoss,
    OptimizerConfig,
    StressHuberLoss,
    TrainingStage,
    ValidationConfig,
    default_training_fn,
)
from nvalchemi.training.peft import load_peft_checkpoint_into_model
from nvalchemi.training.peft.lora import (
    LoRAConfig,
    available_lora_wrappers,
    is_lora_layer,
)
from nvalchemi.training.reference_energies import fit_atomic_reference_energies

# Data and output paths
DATA_ROOT = Path("outputs/lpsc_lora")
LPSC_HF_REPO_ID = "ev-tlt/MACE_finetuning_supplementary"
LPSC_HF_FILENAME = "li/data/LPSC_600.extxyz"
REFERENCE_ENERGIES_PATH = DATA_ROOT / "reference_energies" / "medium-mpa-0-lpsc.json"
PEFT_CHECKPOINT_DIR = DATA_ROOT / "peft_checkpoints"
REFERENCE_ENERGY_FIT_KIND = "baseline_residual_added_to_checkpoint_e0"

# Training, loss function, and optimization settings
SEED = 42
VALIDATION_FRACTION = 0.1
BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 64
TRAINING_EPOCHS = 5
VALIDATION_EVERY_STEPS = 50
LEARNING_RATE = 5.0e-4
WEIGHT_DECAY = 1.0e-4
HUBER_DELTA = 0.01
ENERGY_WEIGHT = 1.0
FORCE_WEIGHT = 10.0
STRESS_WEIGHT = 20.0

# Foundation model and LoRA settings
MACE_CHECKPOINT = "medium-mpa-0"
MACE_DTYPE = torch.float32
ENABLE_CUEQ = True
LORA_RANK = 16
LORA_ALPHA = 2.0
LORA_DROPOUT = 0.0
LORA_TARGET_PATTERNS = (
    "main.model.node_embedding.linear",
    "main.model.interactions.0.linear",
    "main.model.interactions.*.linear_up",
    "main.model.interactions.*.conv_tp_weights.layer[123]",
    "main.model.products.*.linear",
    "main.model.readouts.*.linear",
    "main.model.readouts.*.linear_1",
    "main.model.readouts.*.linear_2",
)
TRAINABLE_PATTERNS = (
    "main.model.interactions.1.linear.weight",
    "main.model.interactions.*.density_fn.layer0.weight",
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs used by this example."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(SEED)


# %%
# Downloading and converting LPSC_600 dataset
# -------------------------------------------
# We begin by downloading the raw ``extxyz`` file with ``huggingface_hub``.


def download_lpsc_extxyz() -> Path:
    """Download ``li/data/LPSC_600.extxyz`` from Hugging Face."""
    print(
        f"Downloading {LPSC_HF_FILENAME} from Hugging Face.",
        flush=True,
    )
    return Path(
        hf_hub_download(
            repo_id=LPSC_HF_REPO_ID,
            filename=LPSC_HF_FILENAME,
            repo_type="dataset",
        )
    )


extxyz_path = download_lpsc_extxyz()


# %%
# Once the file is available locally, we read each structure with ASE and convert
# it into :class:`~nvalchemi.data.AtomicData`. The LPSC file stores the labels as
# ``ref_energy`` and ``ref_forces``. ASE treats ``stress`` as a calculator
# result, so we copy it into ``atoms.info`` before calling
# :meth:`~nvalchemi.data.AtomicData.from_atoms`.


def atomic_data_from_atoms(atoms: Any) -> AtomicData:
    """Convert one ASE Atoms object to ALCHEMI AtomicData."""
    if "stress" not in atoms.info and atoms.calc is not None:
        stress = atoms.calc.results.get("stress")
        if stress is not None:
            atoms.info["stress"] = stress
    return AtomicData.from_atoms(
        atoms,
        energy_key="ref_energy",
        forces_key="ref_forces",
        stress_key="stress",
        dtype=MACE_DTYPE,
    )


samples = [atomic_data_from_atoms(atoms) for atoms in iread(extxyz_path, index=":")]
print(f"Loaded {len(samples)} structures into memory.", flush=True)


# %%
# Creating train/validation subsets
# ---------------------------------
# We shuffle once with a fixed seed, then build training and validation splits.

if len(samples) < 2:
    raise ValueError("Need at least two structures for train/validation split.")
random.Random(SEED).shuffle(samples)
validation_count = max(1, int(round(len(samples) * VALIDATION_FRACTION)))
full_batch = Batch.from_data_list(samples)
train_batch = full_batch[:-validation_count]
validation_batch = full_batch[-validation_count:]
del samples, full_batch
print(
    f"Created deterministic split: {train_batch.num_graphs} train, "
    f"{validation_batch.num_graphs} validation.",
    flush=True,
)


# %%
# Loading batches
# ---------------
# The converted LPSC subset is small enough to keep each split in memory. The
# dataloader then yields mini-batches and moves them to the selected device.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    batch: Batch,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a dataloader"""
    return DataLoader(
        InMemoryDataset(
            in_memory_batch=batch,
            device=device,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        prefetch_factor=0,
        num_streams=1,
        use_streams=False,
        pin_memory=True,
    )


train_loader = make_loader(
    train_batch,
    batch_size=BATCH_SIZE,
    shuffle=True,
    device=device,
)
validation_loader = make_loader(
    validation_batch,
    batch_size=VALIDATION_BATCH_SIZE,
    shuffle=False,
    device=device,
)


# %%
# Preparing the fine-tuning model
# -------------------------------
# Before loading the fine-tuning model, we need to refit the per-element
# reference energies.
# ``fit_atomic_reference_energies`` counts how many atoms of each element appear
# in every training structure, then solves a least-squares problem to find the
# per-element energy corrections that best explain the remaining error of
# ``medium-mpa-0``. We add those residual corrections back to the checkpoint's
# original reference energies before loading the fine-tuning model. This follows
# the model-aware E0 reestimation workflow described in Tompa et al.,
# https://arxiv.org/abs/2606.12704.


def baseline_energy_fn(model: MACEWrapper, neighbor_hook: NeighborListHook) -> Any:
    """Return a callable that evaluates baseline graph energies."""

    def baseline(batch: Any) -> torch.Tensor:
        ctx = HookContext(batch=batch, model=model)
        neighbor_hook(ctx, TrainingStage.BEFORE_FORWARD)
        return model(batch)["energy"]

    return baseline


def mace_checkpoint_reference_energies(model: MACEWrapper) -> dict[int, float]:
    """Return the per-element reference energies stored in a MACE checkpoint."""
    atomic_numbers = [
        int(atomic_number)
        for atomic_number in torch.as_tensor(model.model.atomic_numbers)
        .detach()
        .cpu()
        .tolist()
    ]
    atomic_energies = (
        model.model.atomic_energies_fn.atomic_energies.detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )
    if len(atomic_numbers) != len(atomic_energies):
        raise ValueError(
            "MACE atomic number and reference-energy tables have different lengths: "
            f"{len(atomic_numbers)} != {len(atomic_energies)}."
        )
    return {
        atomic_number: float(energy)
        for atomic_number, energy in zip(atomic_numbers, atomic_energies, strict=True)
    }


def add_reference_energy_corrections(
    checkpoint_reference_energies: dict[int, float],
    residual_corrections: dict[int, float],
) -> dict[int, float]:
    """Add fitted residual corrections to checkpoint reference energies."""
    return {
        atomic_number: checkpoint_reference_energies[atomic_number] + correction
        for atomic_number, correction in residual_corrections.items()
    }


if REFERENCE_ENERGIES_PATH.exists():
    # Reuse the cached fit when this example has already been run.
    raw = json.loads(REFERENCE_ENERGIES_PATH.read_text())
    cache_is_current = raw.get("fit_kind") == REFERENCE_ENERGY_FIT_KIND
else:
    raw = {}
    cache_is_current = False

if cache_is_current:
    reference_energies = {
        int(atomic_number): float(energy)
        for atomic_number, energy in raw["reference_energies"].items()
    }
    print(
        f"Loaded cached refitted reference energies for "
        f"{len(reference_energies)} elements.",
        flush=True,
    )
else:
    # Otherwise, fit the corrections from the training structures.
    foundation_model = MACEWrapper.from_checkpoint(
        MACE_CHECKPOINT,
        device=device,
        dtype=MACE_DTYPE,
        enable_cueq=ENABLE_CUEQ,
        compile_model=False,
    )
    foundation_model.model_config.active_outputs = {"energy"}
    foundation_model.eval()
    reference_neighbor_hook = NeighborListHook(
        foundation_model.model_config.neighbor_config,
        stage=TrainingStage.BEFORE_FORWARD,
    )
    checkpoint_reference_energies = mace_checkpoint_reference_energies(foundation_model)
    reference_loader = make_loader(
        train_batch,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )
    try:
        fit_results = fit_atomic_reference_energies(
            reference_loader,
            baseline_energy_fn=baseline_energy_fn(
                foundation_model,
                reference_neighbor_hook,
            ),
            device=device,
            max_batches=None,  # Fit all structures
        )
    finally:
        reference_loader.dataset.close()
        del foundation_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    residual_reference_energy_corrections = fit_results.reference_energies
    reference_energies = add_reference_energy_corrections(
        checkpoint_reference_energies,
        residual_reference_energy_corrections,
    )
    REFERENCE_ENERGIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_ENERGIES_PATH.write_text(
        json.dumps(
            {
                "checkpoint": MACE_CHECKPOINT,
                "fit_kind": REFERENCE_ENERGY_FIT_KIND,
                "reference_energies": reference_energies,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"Fit residual reference-energy corrections for "
        f"{len(reference_energies)} elements "
        f"from {fit_results.num_structures} structures.",
        flush=True,
    )


# %%
# The fine-tuning model is then loaded with the refitted reference energies by
# passing them through ``atomic_energies``.


model = MACEWrapper.from_checkpoint(
    MACE_CHECKPOINT,
    device=device,
    dtype=MACE_DTYPE,
    enable_cueq=ENABLE_CUEQ,
    compile_model=False,
    atomic_energies=reference_energies,
)
model.model_config.active_outputs = {"energy", "forces", "stress"}
model.train()


# %%
# Checking supported LoRA layers
# ------------------------------
# Registered wrappers define which layer classes can receive LoRA adapters.


def class_name(cls: type) -> str:
    """Return a compact import path for a class."""
    return f"{cls.__module__}.{cls.__qualname__}"


# Print currently registered layer-to-LoRA wrapper pairs.
wrappers = available_lora_wrappers()
print(f"Available LoRA wrappers: {len(wrappers)}", flush=True)
for layer_cls, wrapper_cls in wrappers:
    print(
        f"  - {class_name(layer_cls)} -> {class_name(wrapper_cls)}",
        flush=True,
    )


# %%
# Configuring and running LoRA fine-tuning
# ----------------------------------------
# Similar to :class:`~nvalchemi.training.TrainingStrategy` and
# :class:`~nvalchemi.training.FineTuningStrategy`, this workflow combines the
# training objective, validation reporter, neighbor-list hook, optimizer setup,
# and a :class:`~nvalchemi.training.peft.lora.LoRAConfig`. The LoRA target patterns select
# the MACE layers that receive adapters, while the trainable patterns keep a
# small set of base-model parameters trainable. Module patches can also be
# provided to customize model components, but they are not needed in this
# example.

seed_everything(SEED)

loss_fn: ComposedLossFunction = (
    ENERGY_WEIGHT
    * EnergyHuberLoss(
        per_atom=True,
        delta=HUBER_DELTA,
        ignore_nonfinite=True,
    )
    + FORCE_WEIGHT
    * ForceHuberLoss(
        normalize_by_atom_count=False,
        delta=HUBER_DELTA,
        ignore_nonfinite=True,
    )
    + STRESS_WEIGHT
    * StressHuberLoss(
        delta=HUBER_DELTA,
        ignore_nonfinite=True,
    )
)
loss_fn.normalize_weights = False


def validation_scalars(ctx: Any, stage: TrainingStage) -> dict[str, float]:
    """Extract total and unweighted component validation losses for RichReporter."""
    if stage is not TrainingStage.AFTER_VALIDATION or ctx.validation is None:
        return {}

    summary = ctx.validation
    scalars = {"loss": float(summary["total_loss"])}
    scalars.update(
        {
            name: float(value)
            for name, value in summary["per_component_unweighted"].items()
        }
    )
    return scalars


rich_reporter = RichReporter(
    title="LPSC LoRA validation",
    custom_scalars={"validation": validation_scalars},
    include_losses=False,
    include_optimizer_lrs=False,
    layout="training",
    plot_keys=(
        "validation/loss",
        "validation/ForceHuberLoss",
        "validation/EnergyHuberLoss",
        "validation/StressHuberLoss",
    ),
    max_scalars=8,
    max_plots=3,
    transient=False,
)

strategy = FineTuningStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
    ),
    num_epochs=TRAINING_EPOCHS,
    num_steps=None,
    training_fn=default_training_fn,
    loss_fn=loss_fn,
    devices=[device],
    hooks=[
        NeighborListHook(
            model.model_config.neighbor_config,
            stage=TrainingStage.BEFORE_FORWARD,
        ),
        ReportingOrchestrator(
            [rich_reporter],
            stages={TrainingStage.AFTER_VALIDATION},
            rank_zero_only=True,
        ),
    ],
    validation_config=ValidationConfig(
        validation_data=validation_loader,
        validation_fn=default_training_fn,
        loss_fn=loss_fn,
        every_n_steps=VALIDATION_EVERY_STEPS,
        grad_mode="auto",
        use_ema="auto",
        name="validation",
    ),
    peft_config=LoRAConfig(
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_patterns=LORA_TARGET_PATTERNS,
    ),
    trainable_patterns=TRAINABLE_PATTERNS,
)


# %%
# After the strategy attaches adapters, inspect the inserted LoRA modules before
# training.


# Print LoRA modules installed in strategy.
print("LoRA adapters inserted:", flush=True)
for model_name, model in strategy.models.items():
    module_names = [
        name for name, module in model.named_modules() if is_lora_layer(module)
    ]
    if not module_names:
        print(f"  - {model_name}: none", flush=True)
        continue
    for module_name in module_names:
        print(f"  - {model_name}.{module_name}", flush=True)


# %%
# With the strategy configured, run the fine-tuning loop and save the trained
# PEFT weights. ``RichReporter`` reports validation metrics during training.


seed_everything(SEED)
strategy.run(train_loader)

# Save only optimizer-selected parameters plus buffers; the checkpoint metadata
# records how to recreate the LoRA structure on a fresh base model.
checkpoint_index = strategy.save_checkpoint(
    PEFT_CHECKPOINT_DIR,
    save_trainable_state_only=True,
)
print(
    f"Saved PEFT checkpoint {checkpoint_index} to {PEFT_CHECKPOINT_DIR}",
    flush=True,
)
parameter_summary = strategy.trainable_parameter_summary
print(
    f"LoRA adapter parameters: {parameter_summary['lora']['parameter_count']:,}",
    flush=True,
)
print(
    "Extra trainable base parameters: "
    f"{parameter_summary['extra']['parameter_count']:,}",
    flush=True,
)
print(
    f"Total trainable parameters: {parameter_summary['all']['parameter_count']:,}",
    flush=True,
)

del strategy, model
train_loader.dataset.close()
if device.type == "cuda":
    torch.cuda.empty_cache()

# %%
# Loading the trained LoRA checkpoint for inference
# -------------------------------------------------
# The PEFT checkpoint can be attached to a freshly loaded foundation model for
# inference with
# :func:`~nvalchemi.training.peft.load_peft_checkpoint_into_model`. Here, we use a
# held-out validation batch as a simple inference example. For inference after
# training, the LoRA adapters can also be folded directly into the
# strategy-owned model with
# :func:`~nvalchemi.training.peft.lora.merge_lora_into_model`, for example
# ``merge_lora_into_model(strategy.models["main"])``. The code below
# demonstrates the save-and-load workflow instead.

loaded_model = MACEWrapper.from_checkpoint(
    MACE_CHECKPOINT,
    device=device,
    dtype=MACE_DTYPE,
    enable_cueq=ENABLE_CUEQ,
    compile_model=False,
    atomic_energies=reference_energies,
)
loaded_model = load_peft_checkpoint_into_model(
    loaded_model,
    PEFT_CHECKPOINT_DIR,
)
loaded_model.model_config.active_outputs = {"energy", "forces", "stress"}
loaded_model.eval()
for parameter in loaded_model.parameters():
    parameter.requires_grad_(False)

validation_neighbor_hook = NeighborListHook(
    loaded_model.model_config.neighbor_config,
    stage=TrainingStage.BEFORE_FORWARD,
)

# MACE obtains forces and stress by differentiating the energy, so this
# validation pass keeps gradients enabled during the forward call.
batch = next(iter(validation_loader))
with torch.enable_grad():
    ctx = HookContext(batch=batch, model=loaded_model)
    validation_neighbor_hook(ctx, TrainingStage.BEFORE_FORWARD)
    predictions = default_training_fn(loaded_model, batch)

node_counts = batch.num_nodes_per_graph.to(
    device=device,
    dtype=predictions["predicted_energy"].dtype,
)
energy_counts = node_counts.reshape(
    node_counts.shape[0],
    *([1] * (predictions["predicted_energy"].ndim - 1)),
)
target_energy = batch.energy.to(
    device=device,
    dtype=predictions["predicted_energy"].dtype,
)
target_forces = batch.forces.to(
    device=device,
    dtype=predictions["predicted_forces"].dtype,
)
target_stress = batch.stress.to(
    device=device,
    dtype=predictions["predicted_stress"].dtype,
)
validation_mae_energy = (
    (predictions["predicted_energy"] / energy_counts - target_energy / energy_counts)
    .abs()
    .mean()
)
validation_mae_forces = (predictions["predicted_forces"] - target_forces).abs().mean()
validation_mae_stress = (predictions["predicted_stress"] - target_stress).abs().mean()

print(f"Validation MAE for energy per atom: {validation_mae_energy:.6f}")
print(f"Validation MAE for forces: {validation_mae_forces:.6f}")
print(f"Validation MAE for stress: {validation_mae_stress:.6f}")

validation_loader.dataset.close()

# %%
# Exact values depend on hardware, dependency versions, and training settings.
# As a rough point of comparison, one reference run on an NVIDIA L4 GPU with
# ``TRAINING_EPOCHS = 100`` reached:
#
# .. code-block:: text
#
#    Validation MAE for energy per atom: 0.000363
#    Validation MAE for forces: 0.013591
#    Validation MAE for stress: 0.000173
