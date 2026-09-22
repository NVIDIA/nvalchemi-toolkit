<!-- markdownlint-disable MD014 -->

(generative)=

# Generative Models

The NVIDIA ALCHEMI Toolkit provides an inference driver,
{class}`~nvalchemi.gen.generator.AtomisticGenerator`, for composable inference
workflows of any type: whether they are diffusion/flow matching models, GANs,
VAEs, or even non-neural network based generation. The generation workflows
are designed for performance and composability, both within generative
workflows (e.g. the ability to do structure generation followed by
in-painting) as well as the broader `nvalchemi` ecosystem such as pipelining
{doc}`into dynamics <dynamics>`.

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
| {class}`~nvalchemi.gen.generator.AtomisticGenerator` | The driver. Runs the condition → generate → materialize pipeline, fires hooks, streams, and owns sessions |
| {class}`~nvalchemi.gen.generator.GeneratingFunction` | The callable that owns the family-specific sampling procedure (diffusion, GAN, GA, ...) — and the model, when there is one |
| {class}`~nvalchemi.gen.generator.ConditionFunction` | The optional conditioning step: translates the call's inputs into what the generating function consumes |
| {class}`~nvalchemi.gen.stages.GenerationStage` / {class}`~nvalchemi.hooks.GenerationContext` | The hook lifecycle: when hooks fire, and the per-call state they see |
| {class}`~nvalchemi.models.gen.base.GenerativeModelMixin` / {class}`~nvalchemi.models.gen.base.GenerativeModelConfig` | The model side: what a generative model provides, and what it declares |
| {class}`~nvalchemi.gen.pipeline.GenerationPipeline` | Sequential composition of generators and other batch stages (`\|` sugar) |

## The pipeline: condition, generate

Every call to {meth}`~nvalchemi.gen.generator.AtomisticGenerator.sample` runs
the same steps. Conditioning is optional: it runs when the call provides one
(the driver's `condition_func`, or the generating function's own `condition`
attribute). Hook dispatch happens at the
{class}`~nvalchemi.gen.stages.GenerationStage` points of the steps that
actually ran, and the `AFTER_GENERATE` stage fires only on the `Batch` path:

```text
BEFORE_CONDITION    hooks                     (only when conditioning runs)
                    inputs = condition(inputs, num_samples=...)
AFTER_CONDITION     hooks                     (only when conditioning runs)
                    sample = generator_func(inputs, num_samples=..., rng=...)
BEFORE_MAPPING      hooks                     (always; ctx.sample is set)
if sample is a Batch:                           (the output contract)
    ctx.batch = sample
    AFTER_GENERATE  hooks  (filtering = subsetting ctx.batch)
    return ctx.batch
return sample — any other container passes through untouched
```

The pieces carry the semantics:

- **Condition** — translate the call's inputs into whatever the generating
  function consumes. The inputs are whatever the workflow conditions on: an
  existing {class}`~nvalchemi.data.Batch` of structures, another container,
  or `None` for unconditional generation. A tensor-native condition tiles a
  conditioning batch so each graph appears `num_samples` times. There is no
  default condition: a function that needs one provides its own (a
  `condition` attribute on the callable), and a driver-level
  `condition_func` overrides it for the call. Conditioning *prepares* the
  request — it is not a constraint mechanism. Constraints belong in guidance
  inside the generating function, in filtering hooks, or in downstream
  pipeline stages.
- **Generate** — run the family-specific sampling procedure. The
  {class}`~nvalchemi.gen.generator.GeneratingFunction` receives the (possibly
  conditioned) inputs, the draw count, and an optional RNG. The output
  contract is a {class}`~nvalchemi.data.Batch`: the driver fires the
  `AFTER_GENERATE` hooks on it, validates its device and declared
  `produces_fields`, and pipelines can compose it (including driving
  dynamics). Any other container is a fallback with real losses — the raw
  output passes through untouched, but the `AFTER_GENERATE` hooks are skipped
  and it cannot feed dynamics stages. (Inside the function, a
  {class}`~tensordict.TensorDict` is a fine intermediate: it survives
  `torch.compile`; convert it to a `Batch` before returning.) Nothing about
  the family lives in the `AtomisticGenerator` — a GAN does one forward pass,
  a diffusion model integrates a sampler loop, a GA runs a population loop;
  all behind the same signature. The function also *owns the model* when
  there is one: the model reaches the driver inside the function, not as a
  separate driver argument.
One contract governs the output on the `Batch` path: **filtering** at
`AFTER_GENERATE` is graph-level subsetting of `ctx.batch`; note that `Batch`
does not support zero-graph selections today, so a filter that rejects every
graph raises `IndexError`. The function itself may return a zero-graph batch
(built via {meth}`~nvalchemi.data.Batch.empty`) to signal total rejection —
pipelines then skip the remaining stages for that item.

| Stage | When it fires | What hooks do there |
| ------- | --------------- | --------------------- |
| `BEFORE_CONDITION` | Before the condition step, when one runs | Edit or replace `ctx.inputs` |
| `AFTER_CONDITION` | After conditioning, when one runs | Attach conditioning metadata (e.g. text embeddings); replace the conditioned input |
| `BEFORE_MAPPING` | After generation, before materialization — always | Filter or replace `ctx.sample` before paying materialization cost |
| `AFTER_GENERATE` | After materialization, when a mapping ran | Filter or mutate the generated batch |

## Hooks and the generation context

Hooks are the same {class}`~nvalchemi.hooks.Hook` protocol used by dynamics
and training — `stage`, `frequency`, `__call__(ctx, stage)` — only the stage
enum changes. All hooks in one call share a single
{class}`~nvalchemi.hooks.GenerationContext` and mutate it by *replacing*
its fields; the driver re-reads the context after each dispatch, so an early
hook's edit is visible to later steps and hooks. The context carries:

- `inputs` — the call's inputs (editable at `BEFORE_CONDITION`), holding the
  conditioned value after the condition step,
- `sample` — the raw sample the generating function returned (set at
  `BEFORE_MAPPING`),
- `batch` — the generated batch once the function has returned one (and the
  call's inputs when they were a `Batch`),
- `accepted_mask` — which of the call's candidates were accepted, written by
  filtering hooks (the materialization callable never sees the context; it
  signals total rejection by returning a zero-graph `Batch`), mirroring the
  dynamics `converged_mask` convention,
- `intermediates` — scratch space for hook-to-hook state within one call,
- `step_count` — which generation call this is; drives `frequency` gating.

Generation hooks never run dynamics — optimization is a dynamics engine
downstream of the generator (or a pipeline stage, below), not a hook.

## The model side: mixin and config

A generative model inherits
{class}`~nvalchemi.models.gen.base.GenerativeModelMixin` — the non-energy
counterpart to {class}`~nvalchemi.models.base.BaseModelMixin` — and declares
what it is through a
{class}`~nvalchemi.models.gen.base.GenerativeModelConfig` set as
`self.model_config` in `__init__` (enforced at construction).

The mixin surface is deliberately thin: `forward` (raw output for one forward
call), the `model_config` attribute, and `adapt_output` (raw output →
`ModelOutputs`, keyed by `prediction_outputs`). The mixin owns no scheduler,
sampler, guidance, conditioning, or materialization — those belong to the
generating function, which owns the model.

The config is a small frozen declaration with four fields:
`supports_variable_atoms`, the batch fields the model's conditioning reads
(`consumes_fields`; empty means unconditional), the fields its generated
output carries (`produces_fields`), and `prediction_outputs` (the tensor
names a forward returns). The field declarations are what lets a
[pipeline](#chaining-generators) validate stage links at construction: a
generating function picks them up from its model's config, or declares its
own as attributes on the callable.

## Streaming

{meth}`~nvalchemi.gen.generator.AtomisticGenerator.stream` yields results one
call at a time — one `sample()` per conditioning item, or repeated
unconditional draws with no inputs (bounded by `max_batches`). The stream
yields batches exactly as produced, so a consumer counts graphs itself;
retry/resample logic belongs to the consuming loop.
{meth}`~nvalchemi.gen.generator.AtomisticGenerator.__iter__` is thin sugar
over `stream()`:

```python
for batch in gen.stream(None, max_batches=16):
    ...  # feed a dynamics driver
```

The entry point everywhere is
{meth}`~nvalchemi.gen.generator.AtomisticGenerator.sample`; `__call__` is
syntactic sugar for it, mirroring the dynamics engines
(`FusedStage.__call__` delegates to `step()`).

## Sessions: device, streams, RNG, and compile

An `AtomisticGenerator` is a context manager, mirroring the dynamics engines.
The driver resolves its device at construction — the explicit `device` field
first, then the generating function's own `device` attribute — and validates
it against the host (a CUDA device must exist and be in range). Entering a
session with `with gen:`:

- creates a dedicated CUDA stream when the resolved device is CUDA (opt out
  with `dedicated_stream=False`; always a no-op on CPU),
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
gan = AtomisticGenerator(
    generator_func=make_gan_generate(gan_model),
    compile_kwargs={"fullgraph": True},
)

with gan.compile():  # or set compile_generate=True and compile lazily at entry
    for batch in gan.stream(inputs):
        ...
```

(chaining-generators)=

## Chaining generators

Sequential composition mirrors the dynamics `|` sugar: `gen_a | gen_b`
builds a {class}`~nvalchemi.gen.pipeline.GenerationPipeline`, a thin
orchestrator that folds a conditioning input through heterogeneous stages
(generators, dynamics engines, or any `Batch -> Batch` callable):

```python
pipe = gen_a | gen_b | optimizer      # optimizer: a dynamics engine or FusedStage
out = pipe(cond)
out = pipe(cond, stage_kwargs=[None, None, {"n_steps": 200}])
for batch in pipe.stream(conds):
    ...
```

Pipelines are 1→1 per stage: a filter may shrink a batch, nothing fans out,
and should a stage ever yield a zero-graph batch the remaining stages are
skipped for that item (a defensive contract — no current `Batch` operation
produces one). Each `AtomisticGenerator` stage keeps its own hooks and
context.

The fold duck-types its stages: a stage with a `run` method — a dynamics
engine or a fused stage — is driven to completion with
`stage.run(batch, **kwargs)`, so its own hooks fire inside its loop; any
other stage is called as `stage(batch, **kwargs)`. Per-call options are
addressed to stages with `stage_kwargs`: a single mapping stretches across
every stage (for homogeneous pipelines), or a list of one mapping (or
`None`) per stage, length-checked at entry. A dynamics stage must carry its
own exit criterion (convergence or `n_steps`); the fold offers no step
budget of its own.

`pipe.compile(**kwargs)` compiles each `AtomisticGenerator` stage's
generating function, and `with pipe:` runs the fold on one CUDA stream
shared by every stage that follows the `_stream` convention — generator
stages, and dynamics engines or fused stages, which honor a pre-set stream
when their session is entered — so sequential stages serialize on it with
no cross-stream sync.

For construction-time validation, every generator stage declares the batch
fields it reads and carries — `consumes_fields` / `produces_fields`, set on
the `AtomisticGenerator` or defaulted from the generating function's own
attributes (which in turn default from the model's
{class}`~nvalchemi.models.gen.base.GenerativeModelConfig`). If a stage
declares a field that the immediately upstream generator does not produce,
pipeline construction fails fast — the same contract pattern as
`ModelConfig.required_inputs` elsewhere in the toolkit.

## Building your own model

Putting the pieces together: a small learned decoder that generates a
structure from a latent draw. The model is a plain `torch.nn.Module` with the
mixin and config; a *factory* builds the generating function as a callable
object that owns the model and declares the field contract:

```python
import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
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
            supports_variable_atoms=False,
            consumes_fields=frozenset(),  # unconditional
            produces_fields=frozenset({"positions", "atomic_numbers"}),
            prediction_outputs=("positions",),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw forward: latent -> flat positions."""
        return self.net(x)


class ToyGenerate:
    """The generating function as an object: owns the model, declares fields."""

    def __init__(self, model: ToyDecoder) -> None:
        self.model = model
        self.consumes_fields = model.model_config.consumes_fields
        self.produces_fields = model.model_config.produces_fields

    def __call__(self, inputs=None, *, num_samples=1, rng=None, **kwargs):
        z = torch.randn(num_samples, self.model.latent_dim, generator=rng)
        positions = self.model(z).reshape(num_samples, self.model.num_atoms, 3)
        numbers = torch.full((self.model.num_atoms,), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )


def make_toy_generate(model: ToyDecoder) -> ToyGenerate:
    """Factory: the spec-able pattern — bind the model and config here."""
    return ToyGenerate(model)
```

Note what the model does *not* define: no `condition` (this model is
unconditional), no scheduler or sampler state. Materialization happens inside
the function — one graph per draw — so the driver takes the `Batch` path.

For testing and debugging, the toolkit ships ready-made placeholders —
{class}`~nvalchemi.models.gen.demo.DemoGANModel` and
{class}`~nvalchemi.models.gen.demo.DemoDiffusionModel` — with their own
factories ({func}`~nvalchemi.models.gen.demo.make_demo_gan_generate`,
{func}`~nvalchemi.models.gen.demo.make_demo_diffusion_generate`), plus
{func}`~nvalchemi.models.gen.demo.demo_nonparametric_generation`, a
synthetic-structure source usable standalone or as a pipeline stage.

## Driving it

Wire the procedure into the driver and everything from the first half applies
unchanged — here with a filter hook that drops structures whose largest
displacement from its per-graph centroid exceeds a threshold:

```python
from nvalchemi.gen import AtomisticGenerator, GenerationStage


class MaxDisplacementFilter:
    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
        self.stage = GenerationStage.AFTER_GENERATE
        self.frequency = 1

    def __call__(self, ctx, stage) -> None:
        batch = ctx.batch
        # positions are flattened across graphs; index each atom to its graph
        counts = batch.num_nodes_per_graph
        idx = torch.repeat_interleave(
            torch.arange(batch.num_graphs, device=batch.positions.device), counts
        )
        centroid = torch.zeros(batch.num_graphs, 3, device=batch.positions.device)
        centroid.index_add_(0, idx, batch.positions)
        centroid = centroid / counts[:, None]
        disp = (batch.positions - centroid[idx]).norm(dim=-1)
        max_disp = torch.zeros(batch.num_graphs, device=batch.positions.device)
        max_disp.scatter_reduce_(0, idx, disp, reduce="amax")
        ctx.batch = batch[max_disp <= self.threshold]  # graph-level subsetting


gen = AtomisticGenerator(
    generator_func=make_toy_generate(ToyDecoder(num_atoms=8)),
    hooks=[MaxDisplacementFilter(threshold=2.0)],
)

batch = gen.sample(num_samples=4)   # __call__ works too

for batch in gen.stream(None, max_batches=8):     # stream draws
    ...

with gen.compile(backend="eager"):                # session: stream + RNG + compile
    batch = gen.sample()

pipe = gen | other_generator                      # composed; links validated
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

These components are pieced together inside the generating function.

```python
from typing import Any

import torch
from torch import nn

from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.samplers import sample as pn_sample

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen import AtomisticGenerator, GeneratingFunction


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


class EDMGenerate:
    """The EDM sampling procedure, owning the backbone and the settings."""

    def __init__(
        self,
        model: PositionDenoiser,
        *,
        num_steps: int = 18,
        sigma_max: float = 5.0,
    ) -> None:
        self.model = model
        self.num_steps = num_steps
        self.scheduler = EDMNoiseScheduler(sigma_max=sigma_max)
        self.produces_fields = frozenset({"positions"})

    def __call__(
        self,
        inputs: Any = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Batch:
        # the Predictor: EDM preconditioning wraps the backbone as an
        # x0-predictor, which the scheduler converts into a denoiser
        denoiser = self.scheduler.get_denoiser(x0_predictor=EDMPreconditioner(self.model))
        xN = torch.randn(num_samples, self.model.num_atoms, 3, generator=rng)
        xN = xN * self.scheduler.sigma_max  # EDM: start from noise at sigma_max
        # the sampler: second-order Heun over the schedule's time-steps
        x0 = pn_sample(denoiser, xN, self.scheduler, num_steps=self.num_steps, solver="heun")
        numbers = torch.full((self.model.num_atoms,), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in x0]
        )


def make_edm_generate(
    model: PositionDenoiser, *, num_steps: int = 18, sigma_max: float = 5.0
) -> GeneratingFunction:
    """Piece the EDM components together inside a GeneratingFunction."""
    return EDMGenerate(model, num_steps=num_steps, sigma_max=sigma_max)


diffusion = AtomisticGenerator(
    generator_func=make_edm_generate(PositionDenoiser(num_atoms=32), num_steps=18),
    seed=42,
)

with diffusion:  # session: CUDA stream + seeded RNG
    batch = diffusion.sample(num_samples=16)
```

The backbone is a plain `torch.nn.Module`: the `physicsnemo.diffusion`
interfaces are protocol-based, so anything with the matching call
signature — `forward(x, sigma)` here — slots in without inheriting a
PhysicsNeMo base class; only the diffusion machinery comes from
PhysicsNeMo.

With the deterministic solvers (`"euler"`, `"heun"`), the only randomness
is the initial noise `xN`, which the generating function draws from the
session's `rng` — so `seed` reproduces draws exactly. (The EDM stochastic
solvers inject their own per-step noise.) The optional guidance component
— DPS-style guidance ships under `physicsnemo.diffusion.guidance` —
composes with the predictor before `get_denoiser` converts it to a
denoiser. A conditional variant gives the procedure object a `condition`
attribute that reads fields off the inputs and threads them into the
backbone, e.g. through `class_labels`; and the backbone can additionally
carry `GenerativeModelMixin` with a `GenerativeModelConfig` when it should
validate inside a [GenerationPipeline](#chaining-generators).

### Generative adversarial networks

While GANs may not be as popular as diffusion models, they are relatively
elegant and are excellent pedagogical tools particularly for generation.
A GAN draws noise and runs a single forward pass through the generator
network:

```python
class GANGenerate:
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs=None, *, num_samples=1, rng=None, **kwargs):
        z = torch.randn(num_samples, self.model.latent_dim, generator=rng)
        positions = self.model.decode(z).reshape(num_samples, -1, 3)
        numbers = torch.full((positions.shape[1],), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )


gan = AtomisticGenerator(generator_func=GANGenerate(gan_model))

samples = gan(num_samples=4)              # unconditional
samples = gan(label_batch)                # conditional, via the function's condition
```

### VAE

A VAE samples a latent from the prior and decodes it — same shape as the
GAN, one line different:

```python
def make_vae_generate(model):
    """Bind the model in a factory: the generating function owns it."""

    def vae_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
        z = torch.randn(num_samples, model.latent_dim, generator=rng)
        positions = model.decode(z).reshape(num_samples, -1, 3)
        numbers = torch.full((positions.shape[1],), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )

    return vae_generate
```

## What's next

- {doc}`Dynamics <dynamics>` — relax or run MD on generated structures.
- {doc}`Hooks <hooks>` — the hook protocol in depth.
- {doc}`Training <training>` — train the model side of a generator.
- {doc}`Generative API reference </modules/gen>` — the full class list.

## See also

- {class}`~nvalchemi.gen.generator.AtomisticGenerator`
- {class}`~nvalchemi.gen.generator.GeneratingFunction`
- {class}`~nvalchemi.gen.stages.GenerationStage`
- {class}`~nvalchemi.hooks.GenerationContext`
- {class}`~nvalchemi.gen.pipeline.GenerationPipeline`
