<!-- markdownlint-disable MD014 -->

(distillation_recipes_guide)=

# Reproducible Distillation Recipes

A distillation run is worth reproducing. The teacher is expensive, the student
is a product, and the number that decides whether the student ships comes from
a holdout the run itself never saw. This guide covers the machinery that makes
a run reproducible: the recipe the CLI writes and runs, the spec round trip
behind it, checkpoints that store the teacher's weights once per checkpoint
root, and restarting an interrupted on-policy run. A *recipe* is one JSON file,
validated by `DistillationJobSpec`, that describes a whole distillation run:
where the teacher and the student come from, the data, the output paths, the
strategy, the on-policy segment loop when there is one, and the acceptance bars
`distill evaluate` gates on. The guide closes with a catalog that maps each
distillation objective to the literature it comes from and to the API symbol
that implements it.

```{tip}
**AI coding assistant?** Load the ``nvalchemi-distillation``
{ref}`agent skill <agent_skills>` for concise instructions on strategy setup,
labeling, on-policy configuration, losses, evaluation, and this CLI.
```

For the concepts, such as what a teacher signal is and how the offline and
on-policy loops differ, and for the symbols behind them, see
{ref}`training-distillation-api`. This page assumes you already have a
teacher, a student, and a dataset.

## The recipe lifecycle

One recipe file carries a run from authoring to verdict. The six stages are:

1. **Spec.** `distill init` writes a validated `DistillationJobSpec` scaffold
   at a chosen student size. Edit it as ordinary JSON.
2. **Pre-flight.** `distill spec report` validates the recipe with the same
   helpers the runtime uses. It then renders a *pre-flight card* of what the
   run intends to do, such as the derived teacher signals, the batch
   composition, and the acceptance bars, before a teacher is loaded onto a GPU.
3. **Run.** `distill spec run` builds the teacher, the student, the data, and
   the strategy, then runs it. Errors the strategy's own constructor raises
   surface as CLI errors rather than tracebacks.
4. **Checkpoint.** The `CheckpointHook` that `init` writes into
   `student.hooks` saves periodic checkpoints. The frozen teacher is stored
   *once per checkpoint root*, not once per checkpoint.
5. **Restore.** `distill spec resume` resumes the run. From Python, use
   `DistillationStrategy.load_checkpoint`, or `restore_checkpoint` into a
   constructed strategy. Besides its weights, an on-policy run resumes its
   trajectory, its propagator counter, its initial-structure position, and its
   replay frames.
6. **Evaluate.** `distill evaluate` scores the trained student over the
   recipe's holdout and renders the acceptance report. It can also export the
   report as JSON. It exits non-zero on a missed bar, so a sweep can gate on
   the command. When the recipe's `student.hooks` carry an `EMAHook`, the
   student is gated on the averaged weights that hook trained rather than on
   the live ones, the same weights the run's own validation reads. The line
   above the report names which weights were scored, and the report records
   the same `"ema"` or `"raw"` marker as `StudentEvaluation.weights`.
   `--map-location` names the one device that the student, the teacher, and
   the holdout are all placed on, so a student trained on a GPU can be scored
   on a host that has none.

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
| `on_policy` | The segment loop; required in on-policy mode and read only there |
| `evaluation` | Holdout and the accuracy bars `distill evaluate` gates on |
| `strategy` | The *strategy bundle*: the `DistillationStrategy.to_spec_dict()` output |
| `notes` | Free text rendered in the report |

A *student tier* is a named size template for the student: a width, a depth,
and a radial-basis count that `distill init --tier` writes into the student's
constructor arguments. It never selects an architecture or a model family.
Here is a scaffold at the `small` tier, trimmed to its structure:

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
the validation loader. `init` records it (`--batch-size`, default `8`) rather
than leaving it unset. An unset batch size falls back to a single graph per
batch, and a run sized in `num_steps` would then see only a fraction of the
data it was asked for. The on-policy *mixture*, which draws each training batch
from the replay buffer and the reference dataset, takes its own
`on_policy.batch_size` instead.

The default loss the scaffold writes matches the teacher's energy and forces:
`EnergyMSELoss(target_key="teacher_energy")` plus
`ForceMSELoss(target_key="teacher_forces")`, at weights `1.0` and `10.0`.
`normalize_weights=False` makes those weights literal coefficients. The teacher
signals are *derived* from the `teacher_*` targets rather than declared, so
adding a loss term is all it takes to ask the teacher for another signal. The
`validation` block is built before the strategy and passed to its constructor,
so a validation loss with a `teacher_*` target of its own widens the derived
set the same way. `spec resume` hands the restored strategy the same validation
config, because a strategy given one after construction re-runs neither the
teacher-signal check nor the prediction-key check.

`mode` alone decides which loop runs. A strategy bundle that
`DistillationStrategy.to_spec_dict()` produces in Python for an on-policy run
carries its own `on_policy` and `reference_dataset` entries. Pasting such a
bundle into an `"offline"` recipe is rejected, rather than quietly rebuilding
a segment loop the recipe says it does not run. In on-policy mode, the
top-level `on_policy` block is the one that is built.

Run `distill schema` for the full JSON schema, which is what an editor or a
sweep generator should validate against.

## CLI usage

The group registers on the existing training entry point, beside `train` and
`finetune`:

```bash
nvalchemi-training distill --help
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

The initial-structure store has no default. Omitting the flag exits non-zero
rather than picking a store. `--dataset` names the *reference dataset*: the
store from which the mixture draws its `1 - replay_ratio` reference share. It
cannot stand in for the initial structures, because it carries no `forces`,
and the propagator reads `forces` off the initial batch before the student's
first forward pass. A reference dataset that does carry `energy` or
`forces` of its own is rejected by the strategy at construction, because the
mixture would then zero-fill those targets for every replay row. Point
`--initial-structures` at a store that a dynamics sink or a labeled relaxation
wrote.

Read the pre-flight card `spec report` renders before every run. It shows the
teacher signals the loss implies, the composition of one training batch, the
batch size the training loader draws, the paths that do not exist on disk yet,
a `checkpoint_dir` with no `CheckpointHook` writing into it, a checkpoint root
that already holds a teacher of its own, and the acceptance bars the recipe
records. `spec run` renders the same card first unless `--no-report` is passed.

The report runs the real validation, not a summary of it. An `on_policy` block
is checked against `OnPolicyConfig`'s own field constraints. A `replay_ratio`
above `1`, a `replay_eviction` other than `"fifo"`, a reserved
`weight_sync_frequency`, or a misspelled setting is therefore refused at
`spec report`, before a teacher reaches a device, rather than surfacing as a
traceback at `spec run`. The `initial_structures` block is checked against the
same description that `InitialStructures.from_spec_dict` rebuilds from. A
budget that is not a positive count is refused there too, and so is a
misspelled budget. The misspelling matters most: a source with no budget
declared is unbudgeted, so a budget that reached no field would have run the
whole job unbudgeted.

The report also refuses everything else the recipe settles on its own:

- a step budget below `1`
- a `dataset.format` that no loader builds
- a teacher or student source the CLI could never load, such as a `mace`
  model with neither an id nor a checkpoint, or a `native-checkpoint` with no
  path
- a `replay_ratio` of `0`, which `OnPolicySettings` refuses on its own
- a `replay_ratio` and `batch_size` that leave one mixture source without a
  whole sample in every batch
- an `on_policy.initial_structures` block that names no store
- a `teacher_scorer` block whose `neighbor_list` or `autocast` is spelled in
  a way the scorer does not accept

A rule the strategy or the segment loop owns is not repeated by the report.
An optimizer configured for the teacher or missing for the student, a
`replay_ratio` of `1` beside the reference dataset a recipe always names, a
`replay_device` off the reference dataset's device, and a source that sets
`recycle` with no `fmax` to graduate anything are refused once, at
`spec run`, with the owner's message. Checks that need the models built are
reported there as CLI errors too.

`spec resume` picks an interrupted run back up from its checkpoint directory
and the recipe that started it. The checkpoint carries the models, the
optimizer and scheduler state, the counters, and the on-policy trajectory. The
recipe supplies the runtime hooks and, for an offline run, the dataloader.
`--budget` chooses whose `num_steps` or `num_epochs` size the continued run:
the checkpoint's stored spec by default, or the recipe's with
`--budget recipe`. Raising `num_steps` in the recipe and resuming with
`--budget recipe` therefore extends a finished run. Either way, a disagreement
between the two budgets is reported with both values. A recipe budget that the
checkpoint has already passed, or one in the other unit (epochs against steps),
is refused rather than applied.

`spec resume` needs a checkpoint to exist, and `init` writes the hook that
produces one. A scaffold puts a
{py:class}`~nvalchemi.training.hooks.CheckpointHook` in `student.hooks`. The
hook writes to `<output-dir>/checkpoints`, the same path the scaffold records
as `output.checkpoint_dir`. It saves every `num_steps // 10` steps, or every
step when that would round to less than one:

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

The scaffolded hook sets `save_at_end`, so it writes a terminal checkpoint at
the next index whenever the run finished on a step the interval missed.
`evaluate` therefore scores the weights the run ended with, and a later
`resume` has nothing left to repeat. A hand-written hook is run as declared:
drop `save_at_end` to end on the cadence alone.

Edit the interval like any other field. Hooks are declared here exactly as
they are in {ref}`finetuning_guide` and {ref}`training_guide`. `timestamp` is
the ISO-8601 stamp that every spec carries and `init` fills in. It is a
required field, so a hand-written hook block needs one too. Because the
scaffold writes the hook from the start, the `init` / `spec report` /
`spec run` / `evaluate --student-checkpoint` sequence above runs as written:
`evaluate` is pointed at the directory the run wrote into. `spec report` still
warns when `output.checkpoint_dir` is set and no `CheckpointHook` writes
*into that directory*, which happens when a recipe drops the hook or points it
somewhere else.

### Every option

`distill init` --- authoring:

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode offline\|on-policy` | `offline` | Which loop the recipe describes. `on-policy` writes the segment block and requires `--initial-structures` |
| `--tier` | `small` | Student tier: a size template of width, depth, and radial-basis count only. Any name in the tier registry `DEFAULT_STUDENT_TIERS` (`small`, `base`, and `large` are built in, and `register_student_tier` adds one), checked when the command runs |
| `--tier-kwargs KEY=VALUE` | --- | Constructor argument overriding or extending the tier's template; repeatable. A value that parses as JSON is written as that type (`hidden_dim=96` is an integer, `activation=silu` a string) |
| `--dataset` | *required* | Teacher-labeled training store; the reference dataset under `--mode on-policy` |
| `--output-dir` | *required* | Run output directory, and where the scaffolded `CheckpointHook` writes |
| `--teacher-model` | `mace` | Teacher source family |
| `--teacher-id` | --- | Published teacher id within that family |
| `--teacher-checkpoint` | --- | Teacher checkpoint path. Use it to distill from a fine-tuned teacher rather than from a published id. A teacher source that names neither is refused at `spec report` |
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
| `--json` | off | Print the normalized recipe after the card, with every omitted field set to its default |

`distill spec run` --- execute:

| Option | Default | Meaning |
| --- | --- | --- |
| `--batch-size` | `dataset.batch_size` | Override the training batch size for this run |
| `--shuffle` / `--no-shuffle` | `--shuffle` | Shuffle the offline training loader when no distributed sampler replaces it |
| `--drop-last` | off | Drop the final incomplete batch of the offline training loader |
| `--prefetch-factor` | `2` | Emitted batches fused per backend read |
| `--num-streams` | `4` | CUDA stream count for dataloader prefetching |
| `--pin-memory` | off | Request pinned-memory reads |
| `--use-streams` / `--no-use-streams` | `--use-streams` | CUDA-stream prefetching when CUDA is available |
| `--distributed` / `--no-distributed` | auto when `WORLD_SIZE > 1` | Attach a {py:class}`~nvalchemi.distributed.DistributedManager` and a {py:class}`~nvalchemi.training.hooks.DDPHook` |
| `--ddp-backend nccl\|gloo` | the hook's own default | Process-group backend forwarded to the hook |
| `--map-location` | the recipe's device | Device a checkpoint loads onto |
| `--validation-dataset` | `dataset.validation_path` | Validation store for this run |
| `--validation-every-epochs` / `--validation-every-steps` | the recipe's `validation` cadence | Validation cadence for this run; at most one of the two |
| `--report` / `--no-report` | `--report` | Render the pre-flight card before executing |

The loader and validation options are the training CLI's own, shared with
`train spec run`. An on-policy run builds its loaders from the segment loop
and takes only the validation options.

`distill spec resume` --- continue:

| Option | Default | Meaning |
| --- | --- | --- |
| `--spec` | *required* | Recipe the run started from. It supplies the data and the hooks, which a checkpoint does not carry, and, with `--budget recipe`, the `num_steps`/`num_epochs` that size the continued run |
| `--checkpoint-index` | `-1` | Index within the checkpoint directory to continue from; `-1` is the latest |
| `--budget checkpoint\|recipe` | `checkpoint` | Whose `num_steps`/`num_epochs` size the continued run. A recipe budget below what the checkpoint has completed, or in the other unit, is refused either way |
| loader and validation options | as for `spec run` | `--batch-size`, `--shuffle`, `--drop-last`, `--prefetch-factor`, `--num-streams`, `--pin-memory`, `--use-streams`, `--validation-dataset`, `--validation-every-epochs`, `--validation-every-steps` |
| `--distributed` / `--no-distributed` | auto when `WORLD_SIZE > 1` | As for `spec run` |
| `--ddp-backend nccl\|gloo` | the hook's own default | As for `spec run` |
| `--map-location` | this rank's device when distributed | Device the checkpoint is loaded onto and the restart continues on. The default keeps every rank from staging its weights through rank zero's device |

`distill evaluate` --- gate:

| Option | Default | Meaning |
| --- | --- | --- |
| `--student-checkpoint` | *required* | Native checkpoint directory holding the trained student |
| `--checkpoint-index` | `-1` | Index within it to score; `-1` is the latest, which after a terminal checkpoint is the weights the run ended with |
| `--holdout` | `evaluation.holdout_path` | Override the holdout store the recipe names |
| `--batch-size` | `evaluation.batch_size` | Holdout loader batch size |
| `--prefetch-factor`, `--num-streams`, `--pin-memory`, `--use-streams` | as for `spec run` | Prefetch settings of the holdout loader, which is never shuffled and keeps its last batch |
| `--weights auto\|ema\|raw` | `auto` | Which student weights to score: `auto` reads the EMA average when the recipe declares an {py:class}`~nvalchemi.training.hooks.EMAHook` (a subclass included) and the trained weights otherwise; `ema` fails when the checkpoint holds no average; `raw` scores the trained weights regardless. Recorded as the report's `weights` |
| `--map-location` | `strategy.devices[0]` | The one device the student, the teacher, the holdout, and the errors are placed on |
| `--json-out` | --- | Write the acceptance report as JSON. A non-finite metric is written as the string `"nan"`, `"inf"`, or `"-inf"`, so the file stays readable by a strict JSON parser |

### Execution flags

`spec run` and `spec resume` scale out the way `train spec run` does.
`--distributed` / `--no-distributed` attaches a
{py:class}`~nvalchemi.distributed.DistributedManager` and a
{py:class}`~nvalchemi.training.hooks.DDPHook`, and defaults to on when
`WORLD_SIZE > 1`. `--ddp-backend` chooses the process-group backend forwarded
to the hook.

With a manager attached, the datasets and the validation loader are built on
the rank's own device rather than on `strategy.devices[0]`. An offline recipe
shards like any other training run. An on-policy recipe generates data in
parallel: each rank propagates its own shard of the initial structures and
labels it with its own teacher replica.

### Student size tiers

`--tier` selects a student tier from the registry
{py:data}`~nvalchemi.training.distillation.cli.DEFAULT_STUDENT_TIERS`, which
holds `small`, `base`, and `large` out of the box. A tier is a **size template
and nothing else**. Each tier is a
{py:class}`~nvalchemi.training.distillation.cli.StudentTier` naming a width, a
depth, and a radial-basis count, which `init` writes into
`student.spec.kwargs` for whatever constructor `--student-cls-path` names. A
tier never selects an architecture or a model family, and `student.tier` is
recorded only so that a report and a sweep can say which size a run belongs
to. Point the tier at your own model and edit the numbers freely:
`--tier-kwargs hidden_dim=96` overrides or extends the template at authoring
time. You can also register a template of your own with
{py:func}`~nvalchemi.training.distillation.cli.register_student_tier`, which
refuses a name already taken. `init --tier` checks the name when the command
runs, so a tier that an imported plugin registered is selectable. The
constructor is called with exactly the recorded keyword arguments.

### Acceptance bars a recipe may carry

`distill evaluate` scores the student over the recipe's holdout and does
nothing else. The bars `evaluation.thresholds` accepts are therefore exactly

```python
measured_bars("accuracy", accuracy_quantities=evaluation.quantities)
```

These are the accuracy bars, narrowed to the quantities the recipe compares,
because an accuracy pass fills only the fields of the quantities it was asked
for. Read the bars off
{func}`~nvalchemi.training.distillation.evaluation.measured_bars` rather than
restating a list, so that a bar added to `AcceptanceThresholds` cannot go
silently unrefused. With `"stress"` compared, all four accuracy bars are
available:

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

Drop `"stress"` from `quantities`, and `max_stress_mae` is refused with it.
Narrow to `["energy"]`, and `max_forces_mae` and `min_force_cosine` are refused
too. Any bar outside the accuracy family is refused whatever the quantities
are: `max_energy_drift_per_atom_per_ns`,
`max_energy_drift_per_atom_per_step`, `max_momentum_drift`,
`max_extensivity_error_per_atom`, `max_rdf_jensen_shannon`,
`min_atoms_per_second`, `min_ns_per_day`, and `max_from_scratch_ratio`. Every
refusal happens when the recipe is parsed, by `spec report` as well as by
`evaluate`. The refusal protects the verdict. `build_acceptance_report` fails
a bar that has no measurement behind it rather than skipping it. A recipe
carrying one of these bars could therefore never be accepted, whatever the
student did, because its verdict would rest on a number nobody measured. Those
bars need a propagator and a timestep, a supercell builder, or a second
trained model, and a recipe names none of them.

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
            extensivity=extensivity_error(
                student, state, drop_keys=propagator.bookkeeping_keys()
            ),
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

`StabilityMonitor.metrics()` is a method, not an attribute. It needs at least
two recorded samples at two different steps. Every metric rebuilds from its
own `to_dict` export with `from_dict`. A sweep can therefore measure each
student in its own job, using `distill evaluate --json-out` for the accuracy
half, and form one report at the end. Each export carries the `weights` marker
of the run that wrote it, so the assembled report still says which student was
measured on its averaged weights and which on its live ones.

`--json-out` writes a non-finite metric as the string `"nan"`, `"inf"`, or
`"-inf"`, not as Python's bare `NaN` and `Infinity` tokens. Those tokens are an
extension to JSON, and a strict reader rejects them. The string keeps the
reason a bar failed visible, where `null` would read as a measurement never
taken. Every `from_dict` reads the string back as the float it stood for, so a
report assembled from such exports keeps the failed verdict.

## Teacher checkpoints: stored once per checkpoint root

The teacher is frozen for the whole run, so writing its weights into every
periodic checkpoint duplicates a model that never changes. With a foundation
teacher, that duplication dominates the cost of checkpointing. Instead, a
strategy may declare that one of its models is stored **once per checkpoint
root**, and `DistillationStrategy` declares the teacher.

The first checkpoint written under a root holds the teacher's weights at
`models/teacher/checkpoints/0.pt`. Every later checkpoint writes no teacher
weight file of its own. It records a *model reference* instead: a
`model_references` manifest entry that stands in for the weight file, naming
the index that holds the weights and a fingerprint of them. A run's hundredth
checkpoint therefore costs the student's weights alone:

```json
"model_references": {
  "teacher": {
    "rebuild": "stored",
    "checkpoint_index": 0,
    "fingerprint": {
      "scheme": "sha256-full",
      "num_tensors": 42,
      "num_elements": 4501000,
      "digest": "9f2c..."
    }
  }
}
```

Loading reads the weights back from the index the model reference names,
into the rebuilt teacher or into the live one the caller supplied. It then
checks them against the `fingerprint`: a SHA-256 over every state-dict entry's
name, shape, and dtype together with the raw bytes of all its values, taken in
key order. Every value counts, so a copy differing in a single weight anywhere
is another teacher, and is refused as one. The bytes are hashed as stored, so
the device the weights were loaded on does not change the digest. Precision is
part of the identity: a copy held at `bfloat16` is a different model to the
fingerprint, and is reported as one. Hashing a whole foundation teacher costs
one pass over its weights, moved to the host in chunks rather than copied:
about a third of a second per 100 million `float32` parameters on an H100,
and a quarter of a second for the same state already on the host.
The `scheme` entry names how the digest was taken. A reader compares only a
fingerprint of the scheme it computes itself; one of another scheme is refused
with a `ValueError` naming the remedy, at save and at load, rather than matched
or mistaken for a different copy. A stored copy that was replaced or truncated
raises `ValueError` at load, rather than quietly training a student against a
different teacher.

A restart reads the saved copy, not the teacher's origin, which makes the
checkpoint tree self-contained. A teacher's `checkpoint_spec()` names the
factory call that built it. The checkpoint still writes that spec to
`models/teacher/spec.json` to rebuild the *architecture* from, exactly as it
does for any other model. The spec is not trusted for the weights. A teacher
loaded from an nvalchemi checkpoint (`teacher.model: "native-checkpoint"` in a
recipe, the ordinary way to distill a fine-tuned foundation model) publishes
the spec of whatever it was originally built from. Rebuilding from that spec
alone would restore the wrong weights. Storing the weights once avoids the
problem and costs one copy per checkpoint root.

One root holds one copy. Saving a *different* copy of a declared model into a
root that already holds one raises `ValueError` instead of storing it. The
model reference is root-global, so moving it would point every checkpoint
already written under that root at weights it was not written with. A
second teacher therefore needs its own checkpoint root, which is what a second
run wants anyway. Repair still works: a copy that still matches the
fingerprint is reused while its weight file is on disk, and written again when
that file went missing. That is how a root whose stored weight file went
missing is made whole. The copy lands at the index being written, and every
checkpoint under that root then reads from it.

The copy that counts is the one on disk, not the manifest entry that names it.
A root left behind by a plain `TrainingStrategy` or a
`save_checkpoint(models=...)` call carries the teacher's weights at every
index and no `model_references` at all. The first save that references such a
model reads the latest of those files once to fingerprint it. Continuing that
root with a *different* teacher is then refused exactly as above. Continuing
it with the same teacher references the copy already there rather than
writing a second. The rule holds whoever writes. A
`save_checkpoint(models=...)` call that carries the teacher into a root that
already references one is fingerprinted against it and refused if it differs.
A call that saves without the teacher leaves the entry untouched rather than
orphaning the checkpoints that read it. A root written only by declaring
strategies reads no weights back at save time.

A manifest carrying `model_references` stays at `schema_version` 1, so an
nvalchemi older than this release still reads it. Such a reader loads only the
models that have a weight file at the requested index: the student at any
index, and the teacher only at the index the model reference names. Asking
such a reader for the teacher at any other index fails with
`FileNotFoundError` on the file the model reference stands in for. Upgrade
nvalchemi, or ask the older reader for the stored index.

## Serializable versus runtime-only

`DistillationStrategy.to_spec_dict()` carries a whole on-policy run, including
`on_policy` and `reference_dataset`, as *references* rather than objects.
`OnPolicyConfig.to_spec_dict()` writes the `on_policy` block. Most fields
round-trip. A *runtime-only field* holds a live object that no recipe can
describe. Serializing leaves it out of the block with a warning (a policy
instance on `replay_eviction` is recorded as `"fifo"` instead), and the caller
re-supplies it at construction. The table shows how each field travels:

| Field | How it round-trips |
| --- | --- |
| Every `OnPolicySettings` field (`replay_ratio`, `training_steps_per_segment`, `batch_size`, `generation_steps`, `label_frequency`, `replay_capacity`, `replay_eviction`, `replay_device`, `seed`, `fmax`, `weight_sync_frequency`) | Verbatim |
| `dynamics` | `{"cls_path", "kwargs"}`; the student is rebound at build time. A `torch.dtype` or `torch.device` argument travels as its name (`"float64"`, `"cuda:0"`) and is read back for a constructor annotated to take one |
| `teacher_scorer` | Signal set (built-in names, or custom `TeacherSignal` dicts with `name`, `model_output`, `field`, `level`), `dtype`, `probe_seed`, `neighbor_list` (`"rebuild"` or `"reuse"`), `autocast` (`false`, `null`, `true`, or a floating-point dtype name such as `"bfloat16"`), and the model name `"teacher"`. Another `TeacherScorer` travels as its own `to_spec_dict()` under `scorer_cls`, the class path whose `from_spec_dict()` rebuilds it. Sources use the same {py:class}`~nvalchemi.training.distillation.SpecSerializable` protocol. A scorer with neither method is **refused**, with the remedy in the message |
| `initial_structures` | `{"dataset": {"path", "device"}, "max_atoms", "max_edges", "max_batch_size", "recycle"}` --- the store and the *declared* budgets, never the position. A `MultiDataset` is named by the stores it concatenates, as `{"paths": [...], "device"}`, and `reference_dataset` is named the same way. Another `InitialStructuresSource` travels as its own `to_spec_dict()` under `source_cls`, the class path whose `from_spec_dict()` rebuilds it. A source with neither method is **refused**, with the remedy in the message |
| `convergence_hook` | **Runtime-only**: omitted with a warning |
| `capture_sink`, `replay_admission` | **Runtime-only**: omitted with a warning; a rebuilt loop stages frames in host memory and admits every frame |
| `divergence` | **Runtime-only**: omitted with a warning; a rebuilt loop flags non-finite positions or forces, which is the default predicate |
| A policy instance on `replay_eviction` | **Runtime-only**: recorded as `"fifo"`, silently for a `FIFO` instance and with a warning for any other policy; re-supply the policy at construction |

What stays runtime-only is omitted or recorded by name, never approximated.
`convergence_hook`, `capture_sink`, `replay_admission`, and `divergence` are
omitted with a warning. A policy instance on `replay_eviction` is recorded as
`"fifo"`, with a warning unless it is a `FIFO`. A propagator's live
collaborators are omitted with a warning. What no recipe can name at all is
refused instead: a scorer or source with neither `to_spec_dict` nor
`from_spec_dict`, a dataset holding its samples in memory, and a propagator
no import reaches raise `ValueError` from `OnPolicyConfig.to_spec_dict()`, and
`DistillationStrategy.to_spec_dict()` turns that into a warning and leaves the
whole `on_policy` entry out.

- **The replay and capture collaborators**: `capture_sink`,
  `replay_admission`, `divergence`, and a policy instance on
  `replay_eviction`. Until they are re-supplied at construction, a rebuilt
  loop stages frames in host memory, admits every captured frame, flags
  non-finite positions or forces as divergence, and evicts FIFO. The string
  `"fifo"` is the one eviction a recipe spells.

- **`convergence_hook`.** It is a live
  {py:class}`~nvalchemi.dynamics.base.ConvergenceHook`, and no recipe describes
  one. The `fmax` setting beside it is the scalar spelling of the same
  criterion: a force threshold the config builds a hook from. `fmax` does
  travel, so a run that wants to stay serializable sets `fmax` instead.
  Passing the hook whole warns and drops it from the recipe.
- **A propagator's live collaborators**: hooks, a convergence hook, sinks, and
  a sampler the propagator holds itself. Serializing a propagator that carries
  any of them warns and names them, and a rebuilt propagator starts without
  them. The check reads the *live* propagator rather than the constructor
  arguments it can be introspected for. A collaborator registered after
  construction therefore counts, and so does one on a propagator a recipe
  built: the shortcut below skips the introspection, not the warning. The
  segment loop's own `TeacherLabelHook` is excluded. The loop registers it for
  the length of a run and removes it afterwards, and a rebuilt loop registers
  its own, so naming it would fire at every mid-segment checkpoint and say
  nothing.

  Not every missing collaborator fails silently. A missing neighbor-list hook
  is loud: the model reads its neighbor tensors off the batch, and a batch
  carrying none raises `KeyError` on the first step. The others fail silently.
  Without a convergence hook, a relaxation runs its full `generation_steps`
  instead of graduating converged structures. Without sinks, the frames the
  run was capturing are never written. Without a thermostat or logging hook,
  the trajectory samples the wrong ensemble or goes unrecorded.
- **In-memory datasets.** A recipe references a dataset by the store it reads,
  so an `InMemoryDataset` raises with the fix in the message: write it with
  `label_dataset` and point the recipe at the path.

A propagator that a recipe built round-trips as the `on_policy` block it was
built from. That is a shortcut past the introspection below, not past the
warning above. Any other propagator is introspected, and it round-trips
**only if every constructor argument is readable off a same-named
attribute**. A propagator that normalizes an argument into a private internal
is refused by name, not approximated. Every shipped integrator and optimizer
does this, for example `self._dt_init` for a timestep converted to internal
time units. Rebuilding such a propagator would fall back to the constructor's
own defaults for the arguments it hid, which is a different run. Build it
through `OnPolicyConfig.from_spec_dict`, which keeps the reference it built
from, or re-supply `dynamics` at construction.

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
`reference_dataset` overrides. They restore a run whose datasets live in
memory, or whose propagator carries hooks. An explicitly supplied `on_policy`
outranks the spec's own `on_policy` block. The block is the one description
that cannot be complete, so it never quietly replaces a loop the caller
already holds.

A piece that the recipe cannot describe leaves the whole `on_policy` entry out
of the spec with a warning that says why, rather than writing a recipe that
would rebuild into a different run. A strategy rebuilt from such a spec runs
offline.

```{note}
The settings half of the `on_policy` block is exactly `OnPolicySettings`' own
field set, dumped in JSON mode. A setting added to that class therefore
travels in every recipe without a second list to update. Never add a spec
entry for a field the class does not declare.
```

## Restarting an interrupted run

An offline run restarts the way any `TrainingStrategy` does: weights, optimizer
and scheduler state, counters, and hook state come back, and the resumed run
reaches the weights an unbroken run would have.

An on-policy run needs more, because none of that records the propagator's
position in configuration space. The strategy therefore carries a *restart
bundle* through the checkpoint. The bundle holds four extra things: the live
trajectory batch, the propagator's cumulative step count, the initial
structures' position, and the frames already in the replay buffer. It also
records the settings the run used, as described below. An internal
checkpointable hook carries the bundle, so the checkpoint format does not
change. A run that never generates contributes an empty bundle.

The bundle is enough for an exact continuation with the built-in integrators.
Their Langevin noise is drawn from a counter-based generator keyed on the step
count, so once the batch and the counter are restored, the next step draws the
noise it would have drawn. A propagator that carries internal state of its own
is not continued that exactly. A relaxation optimizer's adaptive history lives
outside the batch. A resumed `FIRE` run therefore re-initializes its timestep,
its mixing coefficient, and its uphill counter from the constructor arguments:
the positions continue, but the acceleration restarts. A run that had climbed
to near `dt_max` takes the same path a fresh relaxation from those positions
would.

Restarting lands on a **segment boundary**. A segment that a checkpoint
interrupted part-way is counted as finished on the way in. Its `AFTER_EPOCH`
hooks never fire, the training batches it had left are *not* replayed, and the
mixture sampler advances past its epoch index instead of redrawing the
reference samples it already trained on. The resumed run opens a **fresh**
segment. Every segment begins by generating, so a checkpoint taken part-way
through a training phase costs one extra generation phase, for frames the
interrupted segment had already generated once. The trajectory is continuous
either way. Only the split between generating and training shifts.

**An exhausted run resumes exhausted.** A relaxation run can graduate its last
trajectory and have no structure left to start a fresh one. Its restart bundle
then carries the replay frames and the exhaustion itself rather than a
trajectory. The resumed run keeps training on the buffer without serving
relaxed structures again.

**The initial structures' position comes back with the trajectory.**
`InitialStructures` serves each structure once, and a run that graduates
converged trajectories keeps drawing from it. A restart that reopened the
source at the front of the shard would backfill with structures the
interrupted run had already relaxed. The restart bundle therefore carries the
next row, its wrap count, and the next `system_id`, plus the rank and world
size they were counted in. A position counts the rows one rank owns, so a
position written on another shard is refused rather than misread. The
dataset, the *declared* budgets, and `recycle` are *configuration*, and they
come back from the recipe instead. The rank and the world size come from the
launcher, and belong to neither the position nor the configuration.

The restart bundle also records the settings the run used. Nothing restores
them from the bundle. They let a resumed run whose `OnPolicyConfig` sets one
differently, such as a wider `label_frequency` or a smaller `replay_capacity`,
say so with a `UserWarning` naming the keys. Without them, the run would
silently produce two halves generated under different settings.

Two further properties of the restart bundle are worth budgeting for.

**It is rank-local.** The bundle rides in a strategy checkpoint, which
`CheckpointHook` writes on rank zero alone. It therefore holds one rank's
trajectory and one rank's replay frames. It is consumed only when a single
rank wrote it and a single rank is restoring it. The `on_policy.restart`
setting decides what happens otherwise: when restarting on more than one rank,
when restoring onto one rank a bundle written on a larger world, or when this
rank's shard cannot take the stored position. The default, `"error"`, refuses
to start and names the reason and the remedy. `"reseed"` drops the bundle with
a `UserWarning` and reseeds each rank from its own share of the initial
structures, with a **cold replay buffer**. Until the first segments refill the
buffer, the mixture is drawn from the reference dataset alone, so budget those
segments as cold. `"resume"` refuses as `"error"` does, and also refuses a
restore that carries no bundle at all, for a run that must never silently
start over. A multi-rank restart is therefore a reseed rather than a resume,
and only when you ask for one. Changing `restart` between the two halves of a
run is not reported as settings drift.

**A restore replaces the replay frames rather than merging them.** The
bundle's frames *are* the buffer as of the checkpoint. The buffer outlives a
`run()` call, so a strategy restored while still holding the frames it
generated would otherwise carry the pre-checkpoint half of them twice.
Duplicates do not reduce diversity, because the mixed loader draws with
replacement. They do skew the weighting toward the stale pre-restart states,
which is exactly backwards for an on-policy loop. They also double the buffer
memory and reach the eviction horizon one restart early.
`ReplayBuffer.clear()` is the public form of the same operation.

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

Under a multi-rank launch, `spec resume` defaults `--map-location` to this
rank's device rather than to the device the checkpoint records. The recorded
device is rank zero's: rank zero writes `strategy.json` after `DDPHook` has
collapsed `devices` to the one GPU it pinned. Loading every rank's weights
onto that device would stage the whole world's restore through one
accelerator's memory.
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

Restoring into a strategy you constructed yourself uses the `devices` you
built it with. The checkpoint is staged through `map_location`, which defaults
to the strategy's own primary device. Every model and optimizer state is then
moved onto `devices`, so the two cannot disagree.

```python
strategy = DistillationStrategy(
    ..., devices=[torch.device(f"cuda:{local_rank}")]
)
strategy.restore_checkpoint(run_dir / "checkpoints")
```

## Objective, literature, API

Each distillation objective supported here matches one teacher signal with one
loss term. The literature column names public work the objective comes from,
for orientation. The objectives are not implementations of a specific paper.

| Objective | Teacher signal | Public literature | API |
| --- | --- | --- | --- |
| Total energy matching | `energy` → `teacher_energy` | Hinton, Vinyals & Dean 2015 (response-based knowledge distillation); Behler & Parrinello 2007 (energy-fitted MLIPs) | {py:class}`~nvalchemi.training.losses.EnergyMSELoss`, {py:class}`~nvalchemi.training.losses.EnergyMAELoss`, {py:class}`~nvalchemi.training.losses.EnergyHuberLoss` with `target_key="teacher_energy"` |
| Force matching | `forces` → `teacher_forces` | Ercolessi & Adams 1994 (the force-matching method); Czarnecki et al. 2017 (Sobolev training --- fitting a teacher's derivatives, not only its values) | {py:class}`~nvalchemi.training.losses.ForceMSELoss`, {py:class}`~nvalchemi.training.losses.ForceHuberLoss`, {py:class}`~nvalchemi.training.losses.ForceL2NormLoss` with `target_key="teacher_forces"` |
| Stress / virial matching | `stress` → `teacher_stress` | Thompson et al. 2015 (virial-fitted MLIPs) | {py:class}`~nvalchemi.training.losses.StressMSELoss`, {py:class}`~nvalchemi.training.losses.StressHuberLoss` with `target_key="teacher_stress"` |
| Per-atom energy decomposition | `atomic_energies` → `teacher_atomic_energies` | Behler & Parrinello 2007 (atomic energy decomposition); Romero et al. 2015 (FitNets --- supervising a student on a teacher's intermediate targets) | {py:class}`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss` |
| Representation matching | `embeddings` → `teacher_node_embeddings` | Romero et al. 2015 (FitNets --- regressing a teacher's hidden representation through a learned projection) | {py:class}`~nvalchemi.training.distillation.EmbeddingMatchingLoss` with {py:class}`~nvalchemi.training.distillation.EmbeddingProjector` and {py:func}`~nvalchemi.training.distillation.embedding_distillation_fn` |
| Curvature matching | `hessian` → `teacher_hvp`, `teacher_hvp_probe` | Czarnecki et al. 2017 (Sobolev training, taken here to second order); Hutchinson 1990 (the stochastic probe that makes a curvature term affordable) | {py:class}`~nvalchemi.training.distillation.HessianMatchingLoss` with {py:func}`~nvalchemi.training.distillation.hessian_distillation_fn` |
| Boltzmann matching | `energy` → `teacher_energy`, read as a sample of the student's own Boltzmann distribution | Shell 2008 (relative-entropy minimization between two ensembles); Minka 2005 (the forward/reverse divergence family `beta` interpolates) | {py:class}`~nvalchemi.training.distillation.BoltzmannMatchingLoss` |

The generation side has a literature of its own. Ross, Gordon & Bagnell 2011
(DAgger) argue for training a student on the states its own policy visits,
rather than only on states a reference distribution supplies.
{py:class}`~nvalchemi.training.distillation.OnPolicyConfig` implements that
idea for configuration space.

The last three objectives ask more of the run than a target field.
Representation and curvature matching each need their own `training_fn`
(`embedding_distillation_fn`, `hessian_distillation_fn`), because both sides
of the comparison come from a second student pass. Representation matching
also adds a `"projector"` model with an optimizer entry of its own. Boltzmann
matching requires `on_policy` and refuses a relaxation propagator, because it
reads a batch as a sample of the student's own Boltzmann distribution.
{ref}`training-distillation-api` gives the weighting guidance each one needs.

```{note}
**Extension point.** Adding an objective means a
{py:class}`~nvalchemi.training.losses.BaseLossFunction` subclass whose
`target_key` names a `teacher_*` field, plus a signal in the scorer if the
field is new; the strategy derives the signal set from the loss and needs no
change. Add a row here when you add the term.
```

## See also

- {ref}`training-distillation-api` --- teacher signals, the two loops, and the
  full distillation API reference
- {ref}`training_guide` --- strategies, optimizers, checkpoints
- {ref}`losses_guide` --- composing and weighting loss terms
- {ref}`serialization_guide` --- how specs and checkpoints work in general
