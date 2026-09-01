<!-- markdownlint-disable MD014 -->

(generative)=

# Generative Models

The NVIDIA ALCHEMI Toolkit provides an inference driver,
{class}`~nvalchemi.gen.generator.AtomGenerator`, for composable inference workflows
of any type: whether they are diffusion/flow matching models, GANs, VAEs, or even
non-neural network based generation. The generation workflows are designed for
performance and composability, both within generative workflows (e.g. the ability
to do structure generation followed by in-painting) as well as the broader `nvalchemi`
ecosystem such as pipelining {doc}`into dynamics <dynamics>`.

```{tip}
`nvalchemi` follows a batch-first principle: think and reason about generative
workflows as producing *batches* of structures per call, not one structure at
a time.
```

## The components

The generative API is a small set of pieces with one job each. You meet them
all below; this table is the map.

| Component | Role |
| ----------- | ------ |
| {class}`~nvalchemi.gen.generator.AtomGenerator` | The abstract generation interface. Runs the fixed condition → generate pipeline, fires hooks, streams, and owns sessions |
| {class}`~nvalchemi.gen.generator.GeneratingFunction` | The callable that owns the family-specific sampling procedure (diffusion, GAN, GA, ...) |
| {class}`~nvalchemi.gen.stages.GenerationStage` / {class}`~nvalchemi.hooks.GenerationContext` | The hook lifecycle: when hooks fire, and the per-call state they see |
| {class}`~nvalchemi.models.gen.base.GenerativeModelMixin` / {class}`~nvalchemi.models.gen.base.GenerativeModelConfig` | The model side: what a generative model provides, and what it declares |
| {class}`~nvalchemi.gen.enums.Modality` / {class}`~nvalchemi.gen.enums.GenerativeIntent` | The vocabulary for what a model ingests or emits, and how it is used |
| {class}`~nvalchemi.gen.pipeline.GenerationPipeline` | Sequential composition of generators and other batch stages (`\|` sugar) |

## The two steps: condition and generate

Every call to {meth}`~nvalchemi.gen.generator.AtomGenerator.sample` runs the
same two steps, with hook dispatch at three
{class}`~nvalchemi.gen.stages.GenerationStage` points:

```text
BEFORE_CONDITION    hooks
                    ctx.batch = condition(ctx.cond, num_samples_per_batch)
AFTER_CONDITION     hooks
                    sample = generator_func(model, ..., cond=ctx.batch)
                    ctx.batch = to_batch(sample, ctx.batch)  # materialize
AFTER_GENERATE      hooks  (filtering = subsetting ctx.batch)
return ctx.batch
```

The names carry the semantics:

- **Condition** — turn the conditioning input into the canonical batch for
  this call. `cond` is whatever the model conditions on: an existing
  {class}`~nvalchemi.data.Batch` of structures, another tensor container, or
  `None` for unconditional generation. `condition(cond, num_samples)` builds
  `ctx.batch`; the default ({func}`~nvalchemi.gen.default_condition`) passes
  an already-built batch through, tiled so each conditioning graph appears
  `num_samples` times. A model overrides `condition` to ingest something
  rawer (e.g. a composition string), and a model that is not batch-native at
  all (a genetic algorithm, an MCMC walker) needs no stub — the default
  passes non-batch containers through unchanged.
- **Generate** — run the family-specific sampling procedure. The
  {class}`~nvalchemi.gen.generator.GeneratingFunction` receives the model,
  the draw count, an optional RNG, and `ctx.batch` as `cond`, and returns the
  sample as a {class}`~tensordict.TensorDict` of named tensors (positions,
  atom types, lattice, ...). Nothing about the family lives in the
  `AtomGenerator` — a GAN does one forward pass, a diffusion model integrates
  a sampler loop, a GA runs a population loop; all behind the same signature.
  The generator then *materializes* the sample into a
  {class}`~nvalchemi.data.Batch` — via `output_to_batch_func(sample, batch)`
  or the model's `to_batch` fallback — which replaces `ctx.batch` before
  `AFTER_GENERATE` hooks fire and is what the call returns.

One contract governs the output: **filtering** at `AFTER_GENERATE` is
graph-level subsetting of `ctx.batch`; note that `Batch` does not support
zero-graph selections today, so a filter that rejects every graph raises
`IndexError` (empty-batch semantics would be a separate data-layer change).

| Stage | When it fires | What hooks do there |
| ------- | --------------- | --------------------- |
| `BEFORE_CONDITION` | Before the conditioning batch is built | Edit or replace `ctx.cond` |
| `AFTER_CONDITION` | After conditioning | Attach conditioning metadata (e.g. text embeddings); replace the conditioning batch |
| `AFTER_GENERATE` | After sampling and materialization | Filter or mutate the generated batch |

## Hooks and the generation context

Hooks are the same {class}`~nvalchemi.hooks.Hook` protocol used by dynamics
and training — `stage`, `frequency`, `__call__(ctx, stage)` — only the stage
enum changes. All hooks in one call share a single
{class}`~nvalchemi.hooks.GenerationContext` and mutate it by *replacing*
its fields; the `AtomGenerator` re-reads the context after each dispatch, so
an early hook's edit is visible to later steps and hooks. The context
carries:

- `cond` — the conditioning input (editable at `BEFORE_CONDITION`),
- `batch` — the single canonical batch: built by conditioning, read by the
  generating function, materialized into the generated batch before
  `AFTER_GENERATE` hooks fire.
- `intermediates` — scratch space for hook-to-hook state within one call,
- `step_count` — which generation call this is; drives `frequency` gating.

Generation hooks never run dynamics — optimization is a dynamics engine
downstream of the generator (or a pipeline stage, below), not a hook.

## The model side: mixin, config, and enums

A generative model inherits
{class}`~nvalchemi.models.gen.base.GenerativeModelMixin` — the non-energy
counterpart to {class}`~nvalchemi.models.base.BaseModelMixin` — and declares
what it is through a
{class}`~nvalchemi.models.gen.base.GenerativeModelConfig` set as
`self.model_config` in `__init__` (enforced at construction).

The mixin surface:

- **required**: `forward` (raw output for one forward call) and the
  `model_config` attribute,
- **provided**: `adapt_output` (raw → `ModelOutputs`, keyed by
  `active_prediction_outputs`) and `condition` (delegates to
  {func}`~nvalchemi.gen.default_condition`),
- **optional**: `to_batch(sample, cond_batch)` (materialization fallback) and
  `generate(...)` (generation-source fallback, when no `generator_func` is
  supplied).

The mixin owns no scheduler, sampler, or guidance — family-specific config
belongs in the generating function's closure.

The config describes the model with two enums. `Modality` enumerates the
artifact kinds a model may ingest or emit:

| `Modality` | Artifact |
| ------------ | ---------- |
| `POINT_CLOUD` | Unordered atoms (coordinates + numbers) |
| `GRAPH` | Atomic graph with explicit edges |
| `CRYSTAL` | Atoms plus a lattice/cell and periodicity flags |
| `TEXT` | Text / SMILES / string conditioning |
| `SPECTRA` | One-dimensional spectroscopic or signal data |
| `EMBEDDING` | Dense latent embedding |
| `IMAGE` | Gridded image |

`GenerativeIntent` enumerates the operational roles. Four are
*output-producing* — `CREATE` (from a prior), `SAMPLE` (from a learned
distribution), `PROPOSE` (candidate artifacts), `DECODE` (latent →
artifact) — and four are *input-facing*: `CONDITION`, `COMPLETE`,
`TRANSFORM`, `CONNECT`. The config's `input_modalities` /
`output_modalities` properties are derived from this split.

A `GenerativeModelConfig` then binds intents to modalities
(`intent_modality_map`; every intent must have an entry), names the primary
`output_artifact`, and — always required — declares the batch fields the
model's conditioning reads (`consumes_fields`; empty means unconditional) and
the fields its generated output carries (`produces_fields`). These
declarations are what lets a [pipeline](#chaining-generators) validate stage
links at construction.

## Streaming

{meth}`~nvalchemi.gen.generator.AtomGenerator.stream` yields batches one call
at a time — one `sample()` per conditioning item in `conds`, or repeated
unconditional draws with `conds=None` (bounded by `max_batches`). The stream
yields batches exactly as produced, so a consumer counts graphs itself;
retry/resample logic belongs to the consuming loop.
{meth}`~nvalchemi.gen.generator.AtomGenerator.__iter__` is thin sugar over
`stream()`:

```python
for batch in gen.stream(conds=None, max_batches=16):
    ...  # feed a dynamics driver
```

The entry point everywhere is
{meth}`~nvalchemi.gen.generator.AtomGenerator.sample`; `__call__` is
syntactic sugar for it, mirroring the dynamics engines
(`FusedStage.__call__` delegates to `step()`).

## Sessions: streams, RNG, and compile

An `AtomGenerator` is a context manager, mirroring the dynamics engines.
Entering a session with `with gen:`:

- creates a dedicated CUDA stream when the model is CUDA-resident (a no-op
  on CPU),
- seeds a session-scoped `torch.Generator` from `seed`, advanced per draw
  (outside a session, each call derives `seed + step_count`),
- compiles the generating function when `compile_generate` is set, and
- opens any context-manager hooks (hooks with `__enter__`/`__exit__`).

Exiting unwinds all of it. Compilation wraps the *generating function* only
— conditioning, materialization, and hook dispatch stay eager — and is
best-effort on arbitrary user callables (non-tensor-pure code graph-breaks).
`compile_kwargs` are validated against the installed `torch.compile`
signature at construction and in `compile()`:

```python
gan = AtomGenerator(
    model=gan_model,
    generator_func=gan_generate,
    output_to_batch_func=to_batch,
    compile_kwargs={"fullgraph": True},
)

with gan.compile():  # or set compile_generate=True and compile lazily at entry
    for batch in gan.stream(conds):
        ...
```

(chaining-generators)=

## Chaining generators

Sequential composition mirrors the dynamics `|` sugar: `gen_a | gen_b`
builds a {class}`~nvalchemi.gen.pipeline.GenerationPipeline`, a thin
orchestrator that folds a conditioning input through heterogeneous stages
(generators, dynamics engines, or any `Batch -> Batch` callable):

```python
pipe = gen_a | gen_b | optimizer
out = pipe(cond)
for batch in pipe.stream(conds):
    ...
```

Pipelines are 1→1 per stage: a filter may shrink a batch, nothing fans out,
and should a stage ever yield a zero-graph batch the remaining stages are
skipped for that item (a defensive contract — no current `Batch` operation
produces one). Each `AtomGenerator` stage keeps its own hooks and context.
`pipe.compile(**kwargs)` compiles each `AtomGenerator` stage's generating
function, and `with pipe:` runs the fold on one CUDA stream shared by all
stages — sequential stages serialize on it with no cross-stream sync.

For construction-time validation, every generator stage declares the batch
fields it reads and carries — `consumes_fields` / `produces_fields`, set on
the `AtomGenerator` or defaulted from the model's
{class}`~nvalchemi.models.gen.base.GenerativeModelConfig`. If a stage
declares a field that the immediately upstream generator does not produce,
pipeline construction fails fast — the same contract pattern as
`ModelConfig.required_inputs` elsewhere in the toolkit.

## Building your own model

Putting the pieces together: a small learned decoder that generates a
structure from a latent draw. The model inherits the mixin, declares its
config, and provides `to_batch`; the generating function owns the procedure:

```python
import torch
from tensordict import TensorDict
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen import AtomGenerator, GenerativeIntent, Modality
from nvalchemi.models.gen import GenerativeModelConfig, GenerativeModelMixin


class ToyDecoder(nn.Module, GenerativeModelMixin):
    """Decode a latent draw into a point cloud of ``num_atoms`` atoms."""

    def __init__(self, num_atoms: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atoms = num_atoms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(), nn.Linear(64, num_atoms * 3)
        )
        self.model_config = GenerativeModelConfig(
            intents={GenerativeIntent.CREATE, GenerativeIntent.SAMPLE},
            supports_variable_atoms=False,
            output_artifact=Modality.POINT_CLOUD,
            intent_modality_map={
                GenerativeIntent.CREATE: frozenset({Modality.POINT_CLOUD}),
                GenerativeIntent.SAMPLE: frozenset({Modality.POINT_CLOUD}),
            },
            consumes_fields=frozenset(),  # unconditional
            produces_fields=frozenset({"positions", "atomic_numbers"}),
        )

    def forward(self, data, *, x, **kwargs):
        """Raw forward: latent -> flat positions."""
        return self.net(x)

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None) -> Batch:
        """Materialize: reshape the flat sample into one graph per draw."""
        positions = sample["x1"].reshape(-1, self.num_atoms, 3)
        numbers = torch.full((self.num_atoms,), 6, dtype=torch.long)
        return Batch.from_data_list(
            [
                AtomicData(positions=positions[i], atomic_numbers=numbers)
                for i in range(positions.shape[0])
            ]
        )


def toy_generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
    """The generating function: draw latents, run the decoder."""
    z = torch.randn(num_samples, model.latent_dim, generator=rng)
    return TensorDict({"x1": model.forward(cond, x=z)}, batch_size=[num_samples])
```

Note what the model does *not* define: no `condition` (the default is enough
— this model is unconditional), no scheduler or sampler state.

For testing and debugging, the toolkit ships ready-made placeholders —
{class}`~nvalchemi.models.gen.demo.DemoGANModel` and
{class}`~nvalchemi.models.gen.demo.DemoDiffusionModel` — that run a bare
`AtomGenerator(model=...)` through their own `generate`/`to_batch`
fallbacks, plus
{func}`~nvalchemi.models.gen.demo.demo_nonparametric_generation`, a
synthetic-structure source usable standalone or as a pipeline stage.

## Driving it

Wire the model into the API and everything from the first half applies
unchanged — here with a filter hook that drops structures whose largest
displacement from the mean exceeds a threshold:

```python
from nvalchemi.gen import GenerationStage


class MaxDisplacementFilter:
    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
        self.stage = GenerationStage.AFTER_GENERATE
        self.frequency = 1

    def __call__(self, ctx, stage) -> None:
        centered = ctx.batch.positions - ctx.batch.positions.mean(dim=0)
        keep = centered.norm(dim=-1).max(dim=-1).values <= self.threshold
        ctx.batch = ctx.batch[keep]  # filtering is graph-level subsetting


gen = AtomGenerator(model=ToyDecoder(num_atoms=8), generator_func=toy_generate,
                    hooks=[MaxDisplacementFilter(threshold=2.0)])

batch = gen.sample(num_samples_per_batch=4)   # __call__ works too

for batch in gen.stream(conds=None, max_batches=8):   # stream draws
    ...

with gen.compile(backend="eager"):                    # session: stream + RNG + compile
    batch = gen.sample()

pipe = gen | other_generator                          # composed; links validated
```

## Examples

The PhysicsNeMo diffusion example below shows the integration pattern end
to end; the GAN and VAE sketches after it show that only the generating
function changes across families.

### PhysicsNeMo diffusion

NVIDIA PhysicsNeMo provides an excellent diffusion abstraction which
we make use of here: noise schedulers, preconditioners, and ODE/SDE samplers
behind the `physicsnemo.diffusion` protocols are able to be integrated here
without any adapter code. Models that make use of this interface for
chemistry are forthcoming, and for now we only showcase the interface
with an abstract diffusion model.

We refer interested readers to the [upstream documentation](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/diffusion/introduction.html),
but at a high level the diffusion abstraction comprises:

1. A noise schedule,
2. A sampler,
3. A denoising callable (referred to as a `Predictor`),
4. Optionally, some guidance mechanism

These components are pieced together inside the generator.

```python
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.samplers import sample as pn_sample

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen import AtomGenerator, GeneratingFunction


class PositionDenoiser(nn.Module):  # plain torch: the protocols need no PhysicsNeMo base
    """Stand-in for a trained x0-predictor backbone."""

    def __init__(self, num_atoms: int) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.net = nn.Sequential(
            nn.Linear(num_atoms * 3 + 1, 64),
            nn.SiLU(),
            nn.Linear(64, num_atoms * 3),
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean positions from noisy ones.

        Flattens ``x`` from ``(B, N, 3)``, appends the noise level ``sigma``
        as a per-draw feature, and maps back to ``(B, N, 3)`` through the
        MLP — the x0-prediction the sampler denoises toward.
        ``class_labels`` is accepted for the PhysicsNeMo calling convention
        and unused here.
        """
        b = x.shape[0]
        s = sigma.reshape(b, 1).expand(b, 1)
        return self.net(torch.cat([x.reshape(b, -1), s], dim=-1)).reshape_as(x)

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None) -> Batch:
        """Materialization fallback: one point-cloud graph per draw."""
        numbers = torch.full((sample["x1"].shape[1],), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in sample["x1"]]
        )


def make_edm_generate(
    *, num_steps: int = 18, sigma_max: float = 5.0
) -> GeneratingFunction:
    """Piece the EDM components together inside a GeneratingFunction."""
    # the noise schedule
    scheduler = EDMNoiseScheduler(sigma_max=sigma_max)

    def edm_generate(
        model: PositionDenoiser,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        cond: Batch | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        # the Predictor: EDM preconditioning wraps the backbone as an
        # x0-predictor, which the scheduler converts into a denoiser
        denoiser = scheduler.get_denoiser(x0_predictor=EDMPreconditioner(model))
        xN = torch.randn(num_samples, model.num_atoms, 3, generator=rng)
        xN = xN * sigma_max  # EDM: start from noise at sigma_max
        # the sampler: second-order Heun over the schedule's time-steps
        x0 = pn_sample(denoiser, xN, scheduler, num_steps=num_steps, solver="heun")
        return TensorDict({"x1": x0}, batch_size=[num_samples])

    return edm_generate


diffusion = AtomGenerator(
    model=PositionDenoiser(num_atoms=32),
    generator_func=make_edm_generate(num_steps=18),
    seed=42,
)

with diffusion:  # session: CUDA stream + seeded RNG
    batch = diffusion.sample(num_samples_per_batch=16)
```

The backbone is a plain `torch.nn.Module`: the `physicsnemo.diffusion`
interfaces are protocol-based, so anything with the matching call
signature — `forward(x, sigma)` here — slots in without inheriting a
PhysicsNeMo base class; only the diffusion machinery comes from
PhysicsNeMo. Materialization comes from the model's `to_batch` fallback,
so the wiring needs no `output_to_batch_func`.

With the deterministic solvers (`"euler"`, `"heun"`), the only randomness
is the initial noise `xN`, which the generating function draws from the
session's `rng` — so `seed` reproduces draws exactly. (The EDM stochastic
solvers inject their own per-step noise.) The optional guidance component
— DPS-style guidance ships under `physicsnemo.diffusion.guidance` —
composes with the predictor before `get_denoiser` converts it to a
denoiser. A conditional variant reads fields off `cond` — the
conditioning batch — and threads them into the backbone, e.g. through
`class_labels`; and the backbone can additionally carry
`GenerativeModelMixin` with a `GenerativeModelConfig` (output artifact
`Modality.POINT_CLOUD`) when it should validate inside a
[GenerationPipeline](#chaining-generators).

### Generative adversarial networks

While GANs may not be as popular as diffusion models, they are relatively
elegant and are excellent pedagogical tools particularly for generation.
A GAN draws noise and runs a single forward pass through the generator
network:

```python
def gan_generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
    z = torch.randn(num_samples, model.latent_dim, generator=rng)
    return TensorDict({"x1": model.decode(z)}, batch_size=[num_samples])


gan = AtomGenerator(
    model=gan_model,
    generator_func=gan_generate,
    output_to_batch_func=to_batch,
)

samples = gan(num_samples_per_batch=4)            # unconditional
samples = gan(cond=label_batch)                   # conditional
```

### VAE

A VAE samples a latent from the prior and decodes it — same shape as the
GAN, one line different:

```python
def vae_generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
    z = torch.randn(num_samples, model.latent_dim, generator=rng)
    return TensorDict({"x1": model.decode(z)}, batch_size=[num_samples])
```

## What's next

- {doc}`Dynamics <dynamics>` — relax or run MD on generated structures.
- {doc}`Hooks <hooks>` — the hook protocol in depth.
- {doc}`Training <training>` — train the model side of a generator.
- {doc}`Generative API reference </modules/gen>` — the full class list.

## See also

- {class}`~nvalchemi.gen.generator.AtomGenerator`
- {class}`~nvalchemi.gen.generator.GeneratingFunction`
- {class}`~nvalchemi.gen.stages.GenerationStage`
- {class}`~nvalchemi.hooks.GenerationContext`
- {class}`~nvalchemi.gen.pipeline.GenerationPipeline`
- {func}`~nvalchemi.gen.default_condition`
