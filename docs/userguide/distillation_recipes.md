<!-- markdownlint-disable MD014 -->

(distillation_recipes_guide)=

# Reproducible Distillation Recipes

A distillation run is worth reproducing: the teacher is expensive, the student
is a product, and the number that decides whether the student ships comes from
a holdout the run itself never saw. This guide covers the machinery that makes
a run reproducible --- the JSON recipe the CLI authors and executes, the spec
round trip behind it, checkpoints that store the teacher's weights once per
checkpoint root, and restarting an interrupted on-policy run --- and closes
with the catalog mapping each distillation objective to the literature it comes
from and the API symbol that implements it.

```{tip}
**AI coding assistant?** Load the ``nvalchemi-distillation``
{ref}`agent skill <agent_skills>` for concise instructions on strategy setup,
labeling, on-policy configuration, losses, evaluation, and this CLI.
```

For the concepts --- what a teacher signal is, how the offline and on-policy
loops differ --- see {ref}`distillation_guide`; for the symbols behind them,
see {ref}`training-distillation-api`. This page assumes you already have a
teacher, a student, and a dataset.

## The recipe lifecycle

One recipe file carries a run from authoring to verdict. The six stages are:

1. **Spec.** `distill init` writes a validated `DistillationJobSpec` scaffold
   at a chosen student size. Edit it; it is ordinary JSON.
2. **Pre-flight.** `distill spec report` validates the recipe with the same
   helpers the runtime uses and renders what it intends to do --- derived
   teacher signals, batch composition, acceptance bars --- before a teacher is
   loaded onto a GPU.
3. **Run.** `distill spec run` builds the teacher, the student, the data, and
   the strategy, then runs it. Errors the strategy's own constructor raises
   surface as CLI errors rather than tracebacks.
4. **Checkpoint.** The `CheckpointHook` `init` writes into `student.hooks`
   saves periodic checkpoints. The frozen teacher is stored *once per
   checkpoint root*, not once per checkpoint.
5. **Restore.** `distill spec resume` --- or, from Python,
   `DistillationStrategy.load_checkpoint` or `restore_checkpoint` into a
   constructed strategy --- resumes the run. An on-policy run resumes its
   trajectory, its propagator counter, its structure cursor, and its replay
   frames as well as its weights.
6. **Evaluate.** `distill evaluate` scores the trained student over the
   recipe's holdout, renders the acceptance report, optionally exports it as
   JSON, and exits non-zero on a missed bar so a sweep can gate on the command.
   A recipe whose `student.hooks` carry an `EMAHook` is gated on the averaged
   weights that hook trained rather than on the live ones, the way the run's
   own validation reads them; the line above the report names which weights
   were scored and the report records the same `"ema"` or `"raw"` marker as
   `StudentEvaluation.weights`. `--map-location` names the one device the
   student, the teacher, and the holdout are all placed on, so a student
   trained on a GPU can be scored on a host that has none.

## The recipe file

`DistillationJobSpec` is the pydantic envelope every command reads. It forbids
unknown keys, so a typo is an error rather than a silently ignored setting.

| Member | Meaning |
| --- | --- |
| `mode` | `"offline"` over a teacher-labeled store, or `"on-policy"` |
| `teacher` | `SourceSpec`: where the frozen teacher comes from |
| `student` | `StudentSpec`: a constructor `spec`, or a `source` checkpoint |
| `dataset` | Training store --- the labeled dataset offline, the reference dataset on-policy |
| `output` | `run_dir`, and the `checkpoint_dir` hooks write under |
| `validation` | Optional validation cadence |
| `on_policy` | Segment-loop recipe; required in, and only read in, on-policy mode |
| `evaluation` | Holdout and the accuracy bars `distill evaluate` gates on |
| `strategy` | The `DistillationStrategy.to_spec_dict()` bundle |
| `notes` | Free text rendered in the report |

A scaffold at the `small` tier, trimmed to its structure:

```json
{
  "name": "small-student-offline-distillation",
  "mode": "offline",
  "teacher": {"model": "mace", "model_id": "small-0b"},
  "student": {
    "tier": "small",
    "spec": {
      "cls_path": "my_package.my_module.MyStudentModel",
      "kwargs": {"hidden_dim": 64, "num_layers": 2, "num_radial": 8}
    }
  },
  "dataset": {
    "path": "data/labeled.zarr",
    "format": "alchemi-zarr",
    "batch_size": 8
  },
  "output": {
    "run_dir": "runs/distill",
    "checkpoint_dir": "runs/distill/checkpoints"
  },
  "strategy": {
    "optimizer_configs": {"student": ["<OptimizerConfig spec>"]},
    "num_epochs": null,
    "num_steps": 1000,
    "devices": ["cuda"],
    "loss_fn_spec": "<ComposedLossFunction spec>",
    "training_fn":
      "nvalchemi.training.distillation.strategy.default_distillation_fn",
    "teacher_signals": null,
    "label_missing": true
  }
}
```

`dataset.batch_size` sizes the offline training loader and, in either mode,
the validation loader; `init` records it (`--batch-size`, default `8`) rather
than leaving it unset, because an unset one falls back to a single graph per
batch and a run sized in `num_steps` would then see a fraction of the data it
was asked for. The on-policy mixture takes its own `on_policy.batch_size`
instead.

The default loss the scaffold writes matches the teacher's energy and forces
--- `EnergyMSELoss(target_key="teacher_energy")` plus
`ForceMSELoss(target_key="teacher_forces")` at weights `1.0` and `10.0`, with
`normalize_weights=False` so those are literal coefficients. The teacher
signals are *derived* from those `teacher_*` targets rather than declared, so
adding a term is all it takes to ask the teacher for another signal.

`mode` decides which loop runs, and it is the only thing that does. The
`strategy` bundle a Python-side `DistillationStrategy.to_spec_dict()` produces
for an on-policy run carries its own `on_policy` and `reference_dataset`
entries; pasting one into an `"offline"` recipe is rejected rather than quietly
rebuilding the segment loop the recipe says it is not running. In on-policy
mode the top-level `on_policy` block is the one that is built.

Run `distill schema` for the full JSON schema, which is what an editor or a
sweep generator should validate against.

## CLI usage

The group registers on the existing training entry point, beside `train` and
`finetune`, and is also installed as a `nvalchemi-distill` alias:

```bash
nvalchemi-training distill --help
nvalchemi-distill --help          # the same group
```

Author, review, run, gate:

```bash
nvalchemi-training distill init \
  --tier small \
  --teacher-model mace --teacher-id small-0b \
  --dataset data/labeled.zarr \
  --holdout-dataset data/holdout.zarr \
  --output-dir runs/distill \
  --out recipe.json

nvalchemi-training distill spec report recipe.json
nvalchemi-training distill spec run recipe.json

nvalchemi-training distill spec resume runs/distill/checkpoints \
  --spec recipe.json

nvalchemi-training distill evaluate recipe.json \
  --student-checkpoint runs/distill/checkpoints \
  --json-out acceptance.json
```

`distill init --mode on-policy` additionally writes the segment loop, and
requires `--initial-structures`:

```bash
nvalchemi-training distill init --mode on-policy \
  --teacher-model mace --teacher-id small-0b \
  --dataset data/reference.zarr \
  --initial-structures data/initial_structures.zarr \
  --output-dir runs/onpolicy \
  --out onpolicy.json
```

The initial-structure store has no default, and omitting the flag exits
non-zero rather than picking one. `--dataset` names the *reference dataset*
--- the store the batch mixture draws its `1 - replay_ratio` reference share
from --- and it cannot stand in for the initial structures. It carries no
`forces`, which the propagator reads off the initial batch before the
student's first forward; and one that does carry `energy` or `forces` of its
own is rejected by the strategy at construction, because the mixture would
then zero-fill those targets for every replay row. Point `--initial-structures` at a
store a dynamics sink or a labeled relaxation wrote.

`spec report` is worth reading before every run. It shows the teacher signals
the loss implies, the composition of one training batch, the batch size the
training loader draws, the paths that do not exist on disk yet, a
`checkpoint_dir` with no `CheckpointHook` writing into it, a checkpoint root
that already holds a teacher of its own, and the acceptance bars the recipe
records. `spec run` renders the same card first unless `--no-report` is passed.

Its validation is the real thing rather than a summary of it: an `on_policy`
block is checked against `OnPolicyConfig`'s own field constraints, so a
`replay_ratio` above `1`, a `replay_eviction` other than `"fifo"`, a reserved
`weight_sync_frequency`, or a misspelled setting is refused at `spec report` ---
before a teacher reaches a device --- rather than surfacing as a traceback at
`spec run`. The `initial_structures` block is checked against the same description
`InitialStructures.from_spec_dict` rebuilds through, so a budget that is not a
positive count and a budget spelled wrongly are refused there too --- the
misspelling most of all, since a source with no budget declared is unbudgeted,
and a budget that reached no field would have run the whole job that way.
Refused with them is everything else the recipe settles on its own: a step
budget below `1`, a `dataset.format` no loader builds, a teacher or student
source the CLI could never load (a `mace` model with neither an id nor a
checkpoint, a `native-checkpoint` with no path), a `replay_ratio` of `0` ---
which `OnPolicySettings` refuses on its own --- or of `1`, which only a recipe
can refuse because a recipe always names a reference dataset, a `replay_ratio`
and `batch_size` that leave one mixture source without a whole sample of every
batch, an `on_policy.initial_structures` block naming no store or setting
`recycle` with no `fmax` to graduate anything, and a `replay_device` that is
not the device the reference dataset is loaded on. What still needs the models
built is reported as a CLI error when they are.

`spec resume` picks an interrupted run back up from its checkpoint directory
and the recipe that started it. The checkpoint carries the models, optimizer
and scheduler state, counters, and the on-policy trajectory; the recipe
supplies the runtime hooks and, offline, the dataloader.

It needs a checkpoint to exist, and `init` writes the hook that produces one. A
scaffold puts a {py:class}`~nvalchemi.training.hooks.CheckpointHook` in
`student.hooks`, pointed at `<output-dir>/checkpoints` --- the same path it
records as `output.checkpoint_dir` --- and saving every `num_steps // 10` steps,
or every step when that would round to less than one:

```json
"student": {
  "hooks": [
    {
      "spec": {
        "cls_path": "nvalchemi.training.hooks.checkpoint.CheckpointHook",
        "timestamp": "2026-09-04T10:30:46.756916+00:00",
        "checkpoint_dir": "runs/distill/checkpoints",
        "step_interval": 100
      },
      "stages": []
    }
  ]
}
```

A cadence saves nothing at training end, so `spec run` and `spec resume` write
a terminal checkpoint at the next index whenever the run finished on a step the
interval missed. `evaluate` therefore scores the weights the run ended with,
and a later `resume` has nothing left to repeat.

Edit the interval like any other field; hooks are declared here exactly as they
are in {ref}`finetuning_guide` and {ref}`training_guide`. `timestamp` is the
ISO-8601 stamp every spec carries and `init` fills in --- it is a required
field, so a hand-written hook block needs one too. Because the hook is
written from the start, the `init` / `spec report` / `spec run` /
`evaluate --student-checkpoint` sequence above runs as it is written: the
directory `evaluate` is pointed at is the one the run wrote into. `spec report`
still warns when `output.checkpoint_dir` is set and no `CheckpointHook` writes
*into that directory*, which is what a recipe that dropped the hook --- or
pointed it somewhere else --- earns.

### Every option

`distill init` --- authoring:

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode offline\|on-policy` | `offline` | Which loop the recipe describes. `on-policy` writes the segment block and requires `--initial-structures` |
| `--tier small\|base\|large` | `small` | Student size template: width, depth, and radial-basis count only |
| `--dataset` | *required* | Teacher-labeled training store; the reference dataset under `--mode on-policy` |
| `--output-dir` | *required* | Run output directory, and where the scaffolded `CheckpointHook` writes |
| `--teacher-model` | `mace` | Teacher source family |
| `--teacher-id` | --- | Published teacher id within that family |
| `--teacher-checkpoint` | --- | Teacher checkpoint path. This is how a recipe distills from a fine-tuned teacher rather than from a published id; a source naming neither is refused at `spec report` |
| `--student-cls-path` | `my_package.my_module.MyStudentModel` | Dotted path of the student constructor the tier sizes. The default is a placeholder: edit it, or `spec run` cannot import a student |
| `--lr` | `0.0001` | Student learning rate |
| `--num-steps` | `1000` | Optimizer steps. The scaffolded checkpoint interval is `max(1, num_steps // 10)`, so this also sets how often the run can be resumed or evaluated |
| `--batch-size` | `8` | Samples per training batch, recorded as `dataset.batch_size` |
| `--device` | `cuda` | Device written to `strategy.devices` |
| `--initial-structures` | --- | Store of initial structures the segment loop starts from; required with `--mode on-policy` |
| `--validation-dataset` | --- | Validation store, written to the recipe's `validation` block |
| `--holdout-dataset` | --- | Acceptance holdout store `distill evaluate` scores against |
| `--out` | stdout | Write the recipe JSON to this file instead of printing it |

`distill schema` --- the JSON schema of a recipe:

| Option | Default | Meaning |
| --- | --- | --- |
| `--out` | stdout | Write the schema JSON to this file instead of printing it |

`distill spec report` --- pre-flight:

| Option | Default | Meaning |
| --- | --- | --- |
| `--json` | off | Print the normalized recipe after the card, which is what a recipe defaulted its omitted fields to |

`distill spec run` --- execute:

| Option | Default | Meaning |
| --- | --- | --- |
| `--distributed` / `--no-distributed` | auto when `WORLD_SIZE > 1` | Attach a {py:class}`~nvalchemi.distributed.DistributedManager` and a {py:class}`~nvalchemi.training.hooks.DDPHook` |
| `--ddp-backend nccl\|gloo` | the hook's own default | Process-group backend forwarded to the hook |
| `--map-location` | the recipe's device | Device a checkpoint loads onto |
| `--report` / `--no-report` | `--report` | Render the pre-flight card before executing |

`distill spec resume` --- continue:

| Option | Default | Meaning |
| --- | --- | --- |
| `--spec` | *required* | Recipe that started the run; it supplies the data and the hook intent a checkpoint deliberately does not carry |
| `--checkpoint-index` | `-1` | Index within the checkpoint directory to continue from; `-1` is the latest |
| `--distributed` / `--no-distributed` | auto when `WORLD_SIZE > 1` | As for `spec run` |
| `--ddp-backend nccl\|gloo` | the hook's own default | As for `spec run` |
| `--map-location` | this rank's device when distributed | Device the checkpoint is loaded onto and the restart continues on; the default keeps every rank from staging its weights through rank zero's |

`distill evaluate` --- gate:

| Option | Default | Meaning |
| --- | --- | --- |
| `--student-checkpoint` | *required* | Native checkpoint directory holding the trained student |
| `--checkpoint-index` | `-1` | Index within it to score; `-1` is the latest, which after a terminal checkpoint is the weights the run ended with |
| `--holdout` | `evaluation.holdout_path` | Override the holdout store the recipe names |
| `--batch-size` | `evaluation.batch_size` | Holdout loader batch size |
| `--map-location` | `strategy.devices[0]` | The one device the student, the teacher, the holdout, and the errors are placed on |
| `--json-out` | --- | Write the acceptance report as JSON. A non-finite metric is written as the string `"nan"`, `"inf"`, or `"-inf"`, so the file stays readable by a strict JSON parser |

### Execution flags

`spec run` and `spec resume` scale out the way `train spec run` does:
`--distributed` / `--no-distributed` attaches a
{py:class}`~nvalchemi.distributed.DistributedManager` and a
{py:class}`~nvalchemi.training.hooks.DDPHook`, defaulting to on when
`WORLD_SIZE > 1`, and `--ddp-backend` chooses the process-group backend
forwarded to the hook.

With a manager attached, the datasets and the validation loader are built on
the rank's own device rather than on `strategy.devices[0]`. An offline recipe
shards like any other training run; an on-policy recipe generates data-parallel,
each rank propagating its own shard of the initial structures and labeling it with its
own teacher replica.

### Student size tiers

`--tier` selects `small`, `base`, or `large`. A tier is a **size template and
nothing else** --- a width, a depth, and a radial-basis count written into
`student.spec.kwargs` for whatever constructor `--student-cls-path` names. It
never selects an architecture or a model family, and `student.tier` is recorded
only so a report and a sweep can say which size a run belongs to. Point the
tier at your own model and edit the numbers freely; the constructor is called
with exactly those keyword arguments.

### Acceptance bars a recipe may carry

`distill evaluate` scores the student over the recipe's holdout and does
nothing else, so the bars `evaluation.thresholds` accepts are exactly

```python
measured_bars("accuracy", accuracy_quantities=evaluation.quantities)
```

--- the accuracy bars, narrowed to the quantities the recipe compares, because
an accuracy pass fills only the fields of the quantities it was asked for. Read
them off {func}`~nvalchemi.training.distillation.evaluation.measured_bars`
rather than restating a list; a bar added to `AcceptanceThresholds` then cannot
go silently unrefused. With `"stress"` compared, all four are available:

```json
"evaluation": {
  "holdout_path": "data/holdout.zarr",
  "targets": "teacher",
  "quantities": ["energy", "forces", "stress"],
  "thresholds": {
    "max_energy_per_atom_mae": 0.005,
    "max_forces_mae": 0.05,
    "max_stress_mae": 0.002,
    "min_force_cosine": 0.99
  }
}
```

Drop `"stress"` from `quantities` and `max_stress_mae` is refused with it;
narrow to `["energy"]` and `max_forces_mae` and `min_force_cosine` go too. Any
bar outside the accuracy family --- `max_energy_drift_per_atom_per_ns`,
`max_energy_drift_per_atom_per_step`, `max_momentum_drift`,
`max_extensivity_error_per_atom`, `max_rdf_jensen_shannon`,
`min_atoms_per_second`, `min_ns_per_day`, or `max_from_scratch_ratio` --- is
refused whatever the quantities are. Every refusal lands when the recipe is
parsed, by `spec report` as much as by `evaluate`. The refusal is not tidiness.
`build_acceptance_report` fails a bar that has no measurement behind it rather
than skipping it, so a recipe carrying one of these could never be accepted
whatever the student did: the run would end in a verdict formed against a
number nobody took. Those bars need a propagator and a timestep, a supercell
builder, or a second trained model, and a recipe names none of them.

Measure them in Python instead, and assemble one report at the end:

```python
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StabilityMonitor,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
    extensivity_error,
    measure_throughput,
)

monitor = StabilityMonitor(timestep_fs=0.5, warmup_steps=200)
propagator.register_hook(monitor)
state = propagator.run(seed_batch, n_steps=2000)

report = build_acceptance_report(
    [
        StudentEvaluation(
            name="small",
            accuracy=evaluate_accuracy(
                student, holdout, targets="teacher", scorer=teacher
            ),
            stability=monitor.metrics(),
            throughput=measure_throughput(propagator, state, timestep_fs=0.5),
            extensivity=extensivity_error(student, state),
        )
    ],
    AcceptanceThresholds(
        max_forces_mae=0.05,
        max_energy_drift_per_atom_per_ns=0.005,
        min_ns_per_day=10.0,
        max_extensivity_error_per_atom=1e-4,
    ),
)
print(report.accepted)
```

`StabilityMonitor.metrics()` is a method, not an attribute, and it needs at
least two recorded samples at two different steps. Every metric rebuilds from
its own `to_dict` export with `from_dict`, so a sweep can measure each student
in its own job --- `distill evaluate --json-out` for the accuracy half --- and
form one report at the end. Each export carries the `weights` marker of the
run that wrote it, so the assembled report still says which student was
measured on its averaged weights and which on its live ones.

`--json-out` writes a non-finite metric as the string `"nan"`, `"inf"`, or
`"-inf"` rather than as Python's bare `NaN` and `Infinity` tokens, which are an
extension to JSON that a strict reader rejects. The string keeps the reason a
bar failed visible, where `null` would read as a measurement never taken, and
every `from_dict` reads it back as the float it stood for, so a report
assembled from such exports keeps the failed verdict.

## Teacher checkpoints: stored once per checkpoint root

The teacher is frozen for the whole run, so writing its weights into every
periodic checkpoint duplicates a model that never changed --- with a foundation
teacher, that duplication dominates the cost of checkpointing. Instead, a
strategy may declare that one of its models is stored **once per checkpoint
root**, and `DistillationStrategy` declares the teacher.

The first checkpoint written under a root holds the teacher's weights at
`models/teacher/checkpoints/0.pt`. Every later checkpoint records a
`model_references` entry naming that index and writes no weight file of its
own, so a run's hundredth checkpoint costs the student's weights alone:

```json
"model_references": {
  "teacher": {
    "rebuild": "stored",
    "checkpoint_index": 0,
    "fingerprint": {
      "num_tensors": 42,
      "num_elements": 4501000,
      "digest": "9f2c..."
    }
  }
}
```

Loading reads the weights back from the index the entry names --- into the
rebuilt teacher, or into the live one the caller supplied --- and checks them
against the `fingerprint`, which hashes each state-dict entry's name, shape,
and dtype together with its values, read at `float64` on the host so the device
the weights were loaded on does not change the digest. How many values depends
on the tensor: one holding at most 4096 of them is hashed whole --- which covers
the per-element tables and the biases a change tends to hide in --- while a
larger one contributes 64 values spanning its whole index range, first and last
included, so no tensor ends in a blind tail. Precision is part of the identity:
a copy held at `bfloat16` is a different model to the fingerprint, and is
reported as one, because widening back to `float64` cannot recover what the
cast rounded off. Sampling the large tensors keeps the cost independent of a
foundation teacher's size; the price is that it identifies a model rather than
validating it, and a change confined to the values between two samples of one
large tensor can slip past. A stored copy that was replaced or truncated raises
`ValueError` at load rather than quietly training a student against a different
teacher.

The saved copy, not the teacher's origin, is what a restart reads --- which is
what makes the checkpoint tree self-contained. A teacher's `checkpoint_spec()`
names the factory call that built it, and the checkpoint still writes that to
`models/teacher/spec.json` to rebuild the *architecture* from, exactly as it
does for any other model. It is not trusted for the weights: a teacher loaded
from an nvalchemi checkpoint --- `teacher.model: "native-checkpoint"` in a
recipe, the ordinary way to distill a fine-tuned foundation model --- publishes
the spec of whatever it was originally built from, and rebuilding from that
alone would restore the wrong weights. Storing them once sidesteps the question
and costs one copy per checkpoint root.

One root holds one copy. Saving a *different* copy of a declared model into a
root that already holds one raises `ValueError` instead of storing it: the
`model_references` entry is root-global, so moving it would repoint every
checkpoint already written under that root at weights they were not written
against. A second teacher therefore needs its own checkpoint root --- which is
what a second run wants anyway. The repair path is untouched: a copy that still
matches the fingerprint is written again freely, which is how a root whose
stored weight file went missing is made whole, at a fresh index every
checkpoint under that root then reads from.

The copy that counts is the one on disk, not the manifest entry naming it. A
root a plain `TrainingStrategy` or a `save_checkpoint(models=...)` call left
behind carries the teacher's weights at every index and no `model_references`
at all; the first save that references such a model reads the latest of those
files once to fingerprint it, so continuing that root with a *different*
teacher is refused exactly as above, and continuing it with the same one
references the copy already there rather than writing a second. The rule holds
whoever writes: a `save_checkpoint(models=...)` that carries the teacher into a
root that already references one is re-fingerprinted against it and refused if
it differs, and one that saves without the teacher leaves the entry untouched
rather than orphaning the checkpoints reading it. A root written only by
declaring strategies reads no weights back at save time at all.

A manifest carrying `model_references` stays at `schema_version` 1, so an
nvalchemi older than this release still reads it --- but only the models it
holds a weight file for at the index asked: the student at any index, the
teacher only at the index the entry names. Asking such a reader for the teacher
at any other index fails with `FileNotFoundError` on the file the reference
stands in for; upgrade nvalchemi, or ask that reader for the stored index.

## Serializable versus runtime-only

`DistillationStrategy.to_spec_dict()` carries a whole on-policy run, including
`on_policy` and `reference_dataset`, as *references* rather than objects.
`OnPolicyConfig.to_spec_dict()` is the piece that does the work:

| Field | How it round-trips |
| --- | --- |
| Every `OnPolicySettings` field (`replay_ratio`, `training_steps_per_segment`, `batch_size`, `generation_steps`, `label_frequency`, `replay_capacity`, `replay_eviction`, `replay_device`, `seed`, `fmax`, `weight_sync_frequency`) | Verbatim |
| `dynamics` | `{"cls_path", "kwargs"}`; the student is rebound at build time. A `torch.dtype` or `torch.device` argument travels as its name (`"float64"`, `"cuda:0"`) and is read back for a constructor annotated to take one |
| `teacher_scorer` | Signal set, `dtype`, `probe_seed`, and the model name `"teacher"` |
| `initial_structures` | `{"dataset": {"path", "device"}, "max_atoms", "max_edges", "max_batch_size", "recycle"}` --- the store and the *declared* budgets, never the cursor. A `MultiDataset` is named by the stores it concatenates, as `{"paths": [...], "device"}`; so is `reference_dataset`. Another `InitialStructuresSource` travels as its own `to_spec_dict()` under `source_cls`, the class path its `from_spec_dict()` is called on; a source with neither method is **refused**, with the remedy in the message |
| `convergence_hook` | **Runtime-only**: omitted with a warning |
| `capture_sink`, `replay_admission` | **Runtime-only**: omitted with a warning; a rebuilt loop stages frames in host memory and admits every frame |
| A policy instance on `replay_eviction` | **Runtime-only**: recorded as `"fifo"` with a warning; re-supply the policy at construction |

Four things stay runtime-only, and all four are omitted rather than
approximated:

- **The replay and capture collaborators** --- `capture_sink`,
  `replay_admission`, and a policy instance on `replay_eviction`. A rebuilt
  loop stages frames in host memory, admits every captured frame, and evicts
  FIFO until they are re-supplied at construction; the string `"fifo"` is the
  one eviction a recipe spells.

- **`convergence_hook`.** It is a live
  {py:class}`~nvalchemi.dynamics.base.ConvergenceHook`, and no recipe describes
  one. The `fmax` setting beside it is the scalar spelling of the same
  criterion --- a force threshold the config builds a hook from --- and it
  does travel, so a run that wants to stay serializable sets that instead.
  Passing the hook whole warns and drops it from the recipe.
- **A propagator's live collaborators** --- hooks, a convergence hook, sinks,
  and a sampler the propagator holds itself. Serializing a propagator carrying
  any of them warns and names them, and a rebuilt propagator starts without
  them. The check reads the *live* propagator rather than the constructor
  arguments it can be introspected for, so a collaborator registered after
  construction counts, and so does one on a propagator a recipe built --- the
  shortcut below skips the introspection, not the warning. The segment loop's
  own `TeacherLabelHook` is excluded: the loop registers it for the length of a
  run and removes it afterwards, and a rebuilt loop registers its own, so
  naming it would fire at every mid-segment checkpoint and say nothing.

  What a rebuild loses is worth separating. A missing neighbor-list hook is
  loud, not silent: the model reads its neighbor tensors off the batch, and a
  batch carrying none raises `KeyError` on the first step. The genuinely silent
  losses are the others --- a convergence hook, so a relaxation runs its full
  `generation_steps` instead of graduating converged structures; sinks, so the
  frames the run was capturing are never written; a thermostat or logging hook,
  so the trajectory samples the wrong ensemble, or goes unrecorded.
- **In-memory datasets.** A recipe references a dataset by the store it reads,
  so an `InMemoryDataset` raises with the fix in the message: write it with
  `label_dataset` and point the recipe at the path.

A propagator that a recipe built round-trips as the recipe it was built from,
which is a shortcut past the introspection below --- not past the warning above.
Any other one is introspected, and it round-trips **only if every constructor
argument is readable off a same-named attribute**. One that normalizes an
argument into a private internal --- `self._dt_init` for a timestep converted
to internal time units, which every shipped integrator and optimizer does ---
is refused by name, not approximated: rebuilding it would fall back to the
constructor's own defaults for the arguments it hid, which is a different run.
Build such a propagator through `OnPolicyConfig.from_spec_dict`, which keeps
the reference it built from, or re-supply `dynamics` at construction.

Rebuilding needs the models supplied, because a recipe names them by role
rather than serializing a second copy:

```python
config = OnPolicyConfig.from_spec_dict(
    recipe, student=student, teacher=teacher
)
strategy = DistillationStrategy.from_spec_dict(
    spec, models={"student": student, "teacher": teacher}
)
```

The stores a recipe names are opened on the spec's `devices[0]` rather than
on the device they were recorded with, so `spec resume --map-location cpu`
reads a GPU-written run's data on the host it now trains on, and so does
`load_checkpoint(..., map_location=...)`.

`DistillationStrategy.from_spec_dict` also accepts `on_policy` and
`reference_dataset` overrides, which is how a run whose datasets live in memory
--- or whose propagator carries hooks --- is restored. An explicitly supplied
`on_policy` outranks the spec's own recipe: a recipe is the only description
that cannot be complete, so a loop the caller is already holding is never
quietly replaced by one.

A piece the recipe cannot describe leaves the whole `on_policy` entry out of
the spec and says why, rather than writing a recipe that would rebuild into a
different run. A strategy rebuilt from such a spec is offline-shaped.

```{note}
The settings half of the recipe is exactly `OnPolicySettings`' own field set,
dumped in JSON mode, so a setting added to that class travels in every recipe
without a second list to update. Never add a spec entry for a field the class does not
declare.
```

## Restarting an interrupted run

An offline run restarts the way any `TrainingStrategy` does: weights, optimizer
and scheduler state, counters, and hook state come back, and the resumed run
reaches the weights an unbroken run would have.

An on-policy run needs more, because the propagator's position in configuration
space is not in any of that. The strategy therefore carries four extra things
through the checkpoint --- the live trajectory batch, the propagator's
cumulative step count, the initial structures' cursor, and the frames already in the
replay buffer --- as an internal checkpointable hook, so no checkpoint-format
change is involved and a run that never generates simply contributes an empty
bundle.

That is enough for an exact continuation with the built-in integrators, whose
Langevin noise is drawn from a counter-based generator keyed on the step count:
restore the batch and the counter and the next step draws the noise it would
have drawn. A propagator carrying internal state of its own is not continued
that far. A relaxation optimizer's adaptive history lives outside the batch, so
a resumed `FIRE` run re-initializes its timestep, its mixing coefficient, and
its uphill counter from the constructor arguments: the positions continue, the
acceleration restarts. A run that had climbed to near `dt_max` therefore takes
the same path a fresh relaxation from those positions would.

Restarting lands on a **segment boundary**. A segment a checkpoint interrupted
part-way is counted as finished on the way in: its `AFTER_EPOCH` hooks never
fire, the training batches it had left are *not* replayed, and the mixture
sampler advances past its epoch index instead of redrawing the reference
samples it already trained on. The resumed run opens a **fresh** segment --- and
since every segment begins by generating, a checkpoint taken part-way through a
training phase costs one extra generation phase, for frames the interrupted
segment had already generated once. The trajectory is continuous either way;
only the generate/train split shifts.

**An exhausted run resumes exhausted.** Once a relaxation run has graduated
its last trajectory and has no structure left to start a fresh one, the bundle
carries the replay frames and the exhaustion itself rather than a trajectory,
so the resumed run keeps training on the buffer without serving relaxed
structures again.

**The structure cursor comes back with the trajectory.** `InitialStructures`
serves each structure once, and a run that graduates converged trajectories
keeps drawing from it, so a restart that reopened the cursor at the front of
the shard would backfill with structures the interrupted run had already
relaxed. The bundle therefore carries the cursor, its wrap count, and the next
`system_id`, plus the rank and world size they were counted in --- a cursor
counts positions in one rank's rows, so one written on another shard is
refused rather than misread. The dataset, the *declared* budgets, and
`recycle` are *configuration* and come back from the recipe instead; the rank
and the world size are launcher facts and belong to neither.

The bundle also records the settings it ran under. Nothing reads them back;
they are there so a resumed run whose `OnPolicyConfig` sets one differently ---
a wider `label_frequency`, a smaller `replay_capacity` --- says so with a
`UserWarning` naming the keys, rather than silently producing a run whose two
halves were generated under different settings.

Two further properties of the restart bundle are worth budgeting for.

**It is rank-local.** The bundle rides in a strategy checkpoint, which
`CheckpointHook` writes on rank zero alone, so it holds one rank's trajectory
and one rank's replay frames. It is consumed only when a single rank wrote it
and a single rank is restoring it. Restarting on more than one rank --- or
restoring onto one rank a bundle written on a larger one --- drops it with a
`UserWarning` and reseeds each rank from its own share of the initial structures, with
a **cold replay buffer**. Until the first segments refill it, the mixture is
drawn from the reference dataset alone, so budget those segments as cold. A
multi-rank restart is therefore a reseed rather than a resume.

**A restore replaces the replay frames rather than merging them.** The bundle's
frames *are* the buffer as of the checkpoint, and the buffer outlives a `run()`
call, so a strategy restored while still holding the frames it generated would
otherwise carry the pre-checkpoint half of them twice. That is not a loss of
diversity --- the mixed loader draws with replacement --- but a weighting skew
toward the stale pre-restart states, which is exactly backwards for an
on-policy loop, on top of double the buffer memory and an eviction horizon
reached one restart early. `ReplayBuffer.clear()` is the public form of the
same operation.

```python
strategy.restore_checkpoint(run_dir / "checkpoints")
strategy.run()
```

From the CLI the same restart is one command, against the recipe the run
started from:

```bash
nvalchemi-training distill spec resume runs/onpolicy/checkpoints \
  --spec onpolicy.json
```

### A multi-rank restart loads onto this rank's device

Under a multi-rank launch `spec resume` defaults `--map-location` to this
rank's device rather than to the device the checkpoint records, which is rank
zero's: rank zero writes `strategy.json` after `DDPHook` has collapsed
`devices` to the one GPU it pinned, and loading every rank's weights onto that
device would stage the whole world's restore through one accelerator's memory.
`--map-location` overrides the default and names the device the continued run
takes. A single-rank restart is unaffected.

From Python, name the device once. Rebuilding the strategy from the checkpoint
takes it as `map_location`, which overrides the recorded `devices` before the
strategy is rebuilt from them:

```python
strategy = DistillationStrategy.load_checkpoint(
    run_dir / "checkpoints", map_location=f"cuda:{local_rank}"
)
```

Restoring into a strategy you constructed yourself takes the `devices` you
built it with: the checkpoint is staged through `map_location` --- the
strategy's own primary device when unset --- and every model and optimizer
state is re-homed onto `devices` afterwards, so the two cannot disagree.

```python
strategy = DistillationStrategy(
    ..., devices=[torch.device(f"cuda:{local_rank}")]
)
strategy.restore_checkpoint(run_dir / "checkpoints")
```

## Objective, literature, API

Each distillation objective supported here matches one teacher signal with one
loss term. The literature column names public work the objective comes from,
for orientation --- these are not implementations of a specific paper.

| Objective | Teacher signal | Public literature | API |
| --- | --- | --- | --- |
| Total energy matching | `energy` → `teacher_energy` | Hinton, Vinyals & Dean 2015 (response-based knowledge distillation); Behler & Parrinello 2007 (energy-fitted MLIPs) | {py:class}`~nvalchemi.training.losses.EnergyMSELoss`, {py:class}`~nvalchemi.training.losses.EnergyMAELoss`, {py:class}`~nvalchemi.training.losses.EnergyHuberLoss` with `target_key="teacher_energy"` |
| Force matching | `forces` → `teacher_forces` | Ercolessi & Adams 1994 (the force-matching method); Czarnecki et al. 2017 (Sobolev training --- fitting a teacher's derivatives, not only its values) | {py:class}`~nvalchemi.training.losses.ForceMSELoss`, {py:class}`~nvalchemi.training.losses.ForceHuberLoss`, {py:class}`~nvalchemi.training.losses.ForceL2NormLoss` with `target_key="teacher_forces"` |
| Stress / virial matching | `stress` → `teacher_stress` | Thompson et al. 2015 (virial-fitted MLIPs) | {py:class}`~nvalchemi.training.losses.StressMSELoss`, {py:class}`~nvalchemi.training.losses.StressHuberLoss` with `target_key="teacher_stress"` |
| Per-atom energy decomposition | `atomic_energies` → `teacher_atomic_energies` | Behler & Parrinello 2007 (atomic energy decomposition); Romero et al. 2015 (FitNets --- supervising a student on a teacher's intermediate targets) | {py:class}`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss` |
| Representation matching | `embeddings` → `teacher_node_embeddings` | Romero et al. 2015 (FitNets --- regressing a teacher's hidden representation through a learned projection) | {py:class}`~nvalchemi.training.distillation.EmbeddingMatchingLoss` with {py:class}`~nvalchemi.training.distillation.EmbeddingProjector` and {py:func}`~nvalchemi.training.distillation.embedding_distillation_fn` |
| Curvature matching | `hessian` → `teacher_hvp`, `teacher_hvp_probe` | Czarnecki et al. 2017 (Sobolev training, taken here to second order); Hutchinson 1990 (the stochastic probe that makes a curvature term affordable) | {py:class}`~nvalchemi.training.distillation.HessianMatchingLoss` with {py:func}`~nvalchemi.training.distillation.hessian_distillation_fn` |
| Boltzmann matching | `energy` → `teacher_energy`, read as a sample of the student's own Boltzmann distribution | Shell 2008 (relative-entropy minimization between two ensembles); Minka 2005 (the forward/reverse divergence family `beta` interpolates) | {py:class}`~nvalchemi.training.distillation.BoltzmannMatchingLoss` |

The generation side has a literature of its own: training a student on the
states its own policy visits, rather than only on states a reference
distribution supplies, is the argument of Ross, Gordon & Bagnell 2011 (DAgger),
and it is what {py:class}`~nvalchemi.training.distillation.OnPolicyConfig`
implements for configuration space.

The last three ask more of the run than a target field. Representation and
curvature matching each need their own `training_fn`
(`embedding_distillation_fn`, `hessian_distillation_fn`) because both sides of
the comparison come from a second student pass, and representation matching
adds a `"projector"` model with an optimizer entry of its own. Boltzmann
matching requires `on_policy` and refuses a relaxation propagator, since it
reads a batch as a sample of the student's own Boltzmann distribution.
{ref}`distillation_guide`
and {ref}`training-distillation-api` carry the weighting guidance each one
needs.

```{note}
**Extension point.** Adding an objective means a
{py:class}`~nvalchemi.training.losses.BaseLossFunction` subclass whose
`target_key` names a `teacher_*` field, plus a signal in the scorer if the
field is new; the strategy derives the signal set from the loss and needs no
change. Add a row here when you add the term.
```

## See also

- {ref}`distillation_guide` --- teacher signals, the two loops, and the concepts
  this page builds on
- {ref}`training-distillation-api` --- the full distillation API reference
- {ref}`training_guide` --- strategies, optimizers, checkpoints
- {ref}`losses_guide` --- composing and weighting loss terms
- {ref}`serialization_guide` --- how specs and checkpoints work in general
