<!-- markdownlint-disable MD014 -->

(generative)=

# Generative Models

`nvalchemi` provides an inference driver,
{class}`~nvalchemi.gen.generator.AtomisticGenerator`, for generative workflows across
any model family. Whether you work with diffusion models, flow matching, generative
adversarial networks (GANs), variational autoencoders (VAEs), or heuristic search
algorithms, the driver provides a unified execution loop. It handles batching, seeds,
devices, CUDA streams, lifecycle hooks, and pipelining into {doc}`dynamics simulations
<dynamics>`.

```{tip}
`nvalchemi` follows a batch-first principle. Generative workflows produce
*batches* of structures per call, not individual structures one at a time.
```

## The mental model

Generative inference separates model execution from workflow orchestration:

1. **You write a generating function.** This callable encapsulates the family-specific
   sampling logic and owns the model.
2. **The driver runs the pipeline.**
   {class}`~nvalchemi.gen.generator.AtomisticGenerator` manages conditioning, device
   placement, random number generation, lifecycle hooks, and validation.
3. **The output is a {class}`~nvalchemi.data.Batch`.** Returning a `Batch` lets hooks
   filter structures and allows immediate composition with {doc}`dynamics engines
   <dynamics>`.

```text
BEFORE_CONDITION   hooks                      (only when conditioning is present)
                   inputs = condition(inputs, num_samples=..., rng=...)
AFTER_CONDITION    hooks                      (only when conditioning is present)
                   sample = generator_func(inputs, num_samples=..., rng=...)
if sample is a Batch:                         (the contract path)
    ctx.batch = sample
    AFTER_GENERATE hooks                      (filtering: ctx.batch = ctx.batch[keep])
    return ctx.batch
return sample                                 (other containers pass through raw)
```

The table below summarizes the core generative components:

| Component | Role |
| --- | --- |
| {class}`~nvalchemi.gen.generator.AtomisticGenerator` | The driver |
| {class}`~nvalchemi.gen.generator.GeneratingFunction` | The sampling callable |
| {class}`~nvalchemi.gen.generator.ConditionFunction` | Optional input transform |
| {class}`~nvalchemi.gen.stages.GenerationStage` | The lifecycle stages |
| {class}`~nvalchemi.hooks.GenerationContext` | Per-call hook state |
| {class}`~nvalchemi.gen.pipeline.GenerationPipeline` | Composition via `\|` |
| {class}`~nvalchemi.models.gen.base.GenerativeModelMixin` | The model-side mixin |
| {class}`~nvalchemi.models.gen.base.GenerativeModelConfig` | Model-side declaration |

## The generating function and output contract

A {class}`~nvalchemi.gen.generator.GeneratingFunction` is any callable that accepts
inputs and returns generated structures. It receives the following arguments:

- `inputs`: Conditioning data, an existing {class}`~nvalchemi.data.Batch`, or `None` for
  unconditional generation.
- `num_samples`: Number of independent draws requested for the call.
- `rng`: An optional `torch.Generator` instance for reproducible draws.
- `**kwargs`: Additional runtime options forwarded from the caller.

The function owns the model, the sampling loop, and data materialization. Returning a
{class}`~nvalchemi.data.Batch` follows the **output contract**:

- The driver runs `AFTER_GENERATE` filtering hooks on `ctx.batch`.
- The driver verifies device residency and declared batch fields.
- Downstream {doc}`dynamics engines <dynamics>` can consume the batch directly.

If a function returns another container, the driver passes it through as raw output. The
driver skips `AFTER_GENERATE` hooks, and downstream dynamics stages will reject the
non-`Batch` output.

To signal total rejection, a generating function can return
{meth}`~nvalchemi.data.Batch.empty`. Downstream pipeline stages then skip execution for
that batch.

Here is a minimal generating function producing random carbon clusters:

```python
import torch
from nvalchemi.data import AtomicData, Batch

def random_cluster_generate(
    inputs=None,
    *,
    num_samples: int = 1,
    rng: torch.Generator | None = None,
    num_atoms: int = 8,
    **kwargs,
) -> Batch:
    device = rng.device if rng is not None else "cpu"
    positions = torch.randn(num_samples, num_atoms, 3, generator=rng, device=device)
    atomic_numbers = torch.full((num_atoms,), 6, dtype=torch.long, device=device)
    return Batch.from_data_list(
        [AtomicData(positions=p, atomic_numbers=atomic_numbers) for p in positions]
    )
```

## Driving generation

Wrap your generating function in {class}`~nvalchemi.gen.generator.AtomisticGenerator` to
run it:

```python
from nvalchemi.gen import AtomisticGenerator

gen = AtomisticGenerator(generator_func=random_cluster_generate, seed=42)

# Generate a batch of 4 structures
batch = gen.sample(num_samples=4)  # gen(num_samples=4) is equivalent
```

### Per-call lifecycle

Every call to {meth}`~nvalchemi.gen.generator.AtomisticGenerator.sample` follows a fixed
sequence:

1. **Resolve conditioning**: If a condition function exists (`condition_func` or
   `generator_func.condition`), the driver runs `BEFORE_CONDITION` hooks, transforms
   `inputs`, and runs `AFTER_CONDITION` hooks.
2. **Execute sampling**: The driver calls `generator_func` with the inputs, the resolved
   `num_samples`, and the active `rng`.
3. **Inspect batch output**: When `generator_func` returns a
   {class}`~nvalchemi.data.Batch`, the driver runs `AFTER_GENERATE` hooks, tests device
   residency, and tests declared batch fields.
4. **Pass through raw output**: Any other return type passes through directly without
   batch tests or `AFTER_GENERATE` hooks.

Calling `gen(inputs, ...)` is direct syntactic sugar for `gen.sample(inputs, ...)`.

### Driver configuration

You configure the driver at initialization. The table below lists the available fields:

| Field | Purpose |
| --- | --- |
| `generator_func` | The generating callable that runs the model. |
| `condition_func` | Optional input transform run before generation. |
| `hooks` | List of hooks that run at generation stages. |
| `num_samples` | Default draw count per call. A call argument overrides it. |
| `seed` | Base seed for random number generation. |
| `device` | Target device, tested at initialization. |
| `dedicated_stream` | Creates a dedicated CUDA stream inside sessions. |
| `enable_inference_mode` | Disables gradient tracking inside sessions. |
| `compile_generate`, `compile_kwargs` | Options to compile with `torch.compile`. |
| `required_inputs`, `outputs` | Declared input and output batch fields. |

Explicit arguments override attributes on `generator_func`. If `device`,
`required_inputs`, or `outputs` are unset, the driver reads them from
`generator_func`. The driver tests `device` and `compile_kwargs` at initialization.
For the full list, see {doc}`the API reference </modules/gen>`.

### Streaming

{meth}`~nvalchemi.gen.generator.AtomisticGenerator.stream` yields batches one at a time
across an iterable of inputs. For unconditional generation, pass `inputs=None` along
with `max_batches`:

```python
for batch in gen.stream(None, max_batches=10, num_samples=4):
    print(f"Generated batch with {batch.num_graphs} structures")
```

Calling `iter(gen)` or `for batch in gen:` creates an unbounded stream. Use
`max_batches` in `stream()` whenever you need a bounded loop.

### Sessions: streams, seeds, and compilation

{class}`~nvalchemi.gen.generator.AtomisticGenerator` acts as a context manager:

```python
with gen:
    first_batch = gen.sample(num_samples=4)
    second_batch = gen.sample(num_samples=4)
```

Bare calls (`gen.sample(...)` or `gen(...)`) are sufficient for quick draws. Outside
a session, each call seeds independently using `seed + step_count`.

Sessions (`with gen:`) manage state across multiple calls. Entering a session
performs four setup actions:

1. **Dedicated CUDA stream**: Enters a private CUDA stream on CUDA devices (disable with
   `dedicated_stream=False`). The stream waits on pending work at entry.
2. **Session-scoped RNG**: Initializes a `torch.Generator` from `seed` that advances
   across draws.
3. **Inference mode**: Enters `torch.inference_mode` when `enable_inference_mode=True`
   (off by default). Enable it when your sampler does not need gradients.
4. **Hook lifecycles**: Calls `__enter__` on context-manager hooks for clean teardown
   on exit.

You can compile the generating function with `torch.compile` via
`gen.compile(**compile_kwargs)` or by setting `compile_generate=True`. Compilation
wraps the function call. Hook dispatch stays eager.

Building a `Batch` runs Python and pydantic code that TorchDynamo cannot trace, so a
function that constructs a `Batch` graph-breaks at construction. For end-to-end
capture, compile the model inside your generating function
(`torch.compile(model, ...)`) and keep the wrapper eager.

### Devices, streams, and compile

The driver resolves its target device at construction time. The driver reads
the explicit `device` field first, and then falls back to the `device` attribute
on the generating function, with the following conditions:

- If you specify a CUDA device, that device must exist
on the host.
- When the generating function returns a
{class}`~nvalchemi.data.Batch`, the batch tensors must reside on the resolved
device (i.e. the function cannot change devices).

A device mismatch raises a `ValueError` during the call. In a pipeline,
each stage resolves its own device, and its outputs pass to the next stage
without automatic device migration.

Using the context manager pattern (i.e. `with gen: ...`), the generation
workflow will run on a CUDA stream unless `dedicated_stream` is set to `False`.
At session entry, the new stream waits for active work
on the current caller stream, so session operations do not race caller tasks. A
pipeline session creates one CUDA stream and shares that stream with every stage
that follows the `_stream` convention. These stages include generators, dynamics
engines, and fused stages. The pipeline shares the stream only when the stage
device matches the pipeline stream device, so mixed-device pipelines do not
share streams across devices.

The shared stream serializes stage execution within a pipeline call without
cross-stream synchronization. One caveat however is that exiting a session
does **not** synchronize the CUDA stream: if you dispatch operations on other
streams, you must synchronize or
enqueue a `wait_stream` call before you consume the results. As the session
overview explains, `compile_generate` and `compile_kwargs` compile the
generating function call. Because batch construction causes graph breaks in
TorchDynamo, you must compile the neural network model inside the generating
function for end-to-end graph capture.

## Input conditioning

Conditioning prepares inputs before generation runs. It converts raw inputs (such as
class labels, compositions, or parent structures) into the representation expected by
the generating function.

You supply conditioning through `condition_func` on the generator, or via a `condition`
attribute on the generating function callable. The driver resolves conditioning by
priority:

1. Explicit `condition_func` passed to
   {class}`~nvalchemi.gen.generator.AtomisticGenerator`.
2. A `condition` attribute on `generator_func`.
3. `None` (unconditional; inputs pass directly to `generator_func`).

The `BEFORE_CONDITION` and `AFTER_CONDITION` hook stages fire only when a condition step
is present.

Conditioning prepares the request; it does not enforce physical constraints on the
output. Output constraints belong in guidance terms inside the sampler, in filtering
hooks, or in downstream relaxation stages.

Here is a common pattern: tiling a conditioning batch so each structure receives
`num_samples` draws:

```python
def tile_condition(inputs, *, num_samples: int = 1, rng=None):
    if isinstance(inputs, Batch):
        idx = torch.arange(inputs.num_graphs).repeat_interleave(num_samples)
        return inputs[idx.to(inputs.device)]
    return inputs
```

## Lifecycle hooks and GenerationContext

Generative workflows use the same {class}`~nvalchemi.hooks.Hook` protocol as
{doc}`dynamics <dynamics>` and {doc}`training <training>`. A hook defines `stage`,
`frequency`, and `__call__(ctx, stage)`.

All hooks in a call share a single {class}`~nvalchemi.hooks.GenerationContext` instance:

| Field | Description |
| --- | --- |
| `ctx.batch` | The generated {class}`~nvalchemi.data.Batch` (or a `Batch` input). |
| `ctx.inputs` | Raw inputs before conditioning, conditioned inputs after. |
| `ctx.sample` | Raw sample returned by the generating function. |
| `ctx.accepted_mask` | Optional boolean tensor indicating accepted candidates. |
| `ctx.intermediates` | Scratch dictionary for passing data between hooks. |
| `ctx.step_count` | Counter of generation calls driving frequency gating. |
| `ctx.global_rank` | Distributed process rank for multi-GPU reporting. |
| `ctx.workflow` | Back-reference to the driving generator. |

Hooks mutate state by replacing context fields. For example, filtering at
`AFTER_GENERATE` subsets `ctx.batch`:

```python
import torch

from nvalchemi.data import Batch
from nvalchemi.gen import GenerationStage


class CentroidSpreadFilter:
    def __init__(self, max_spread: float = 3.0) -> None:
        self.max_spread = max_spread
        self.stage = GenerationStage.AFTER_GENERATE
        self.frequency = 1

    def __call__(self, ctx, stage) -> None:
        batch = ctx.batch
        counts = batch.num_nodes_per_graph
        idx = torch.repeat_interleave(
            torch.arange(batch.num_graphs, device=batch.positions.device), counts
        )
        centroid = torch.zeros(batch.num_graphs, 3, device=batch.positions.device)
        centroid.index_add_(0, idx, batch.positions)
        centroid = centroid / counts[:, None]

        dist = (batch.positions - centroid[idx]).norm(dim=-1)
        max_dist = torch.zeros(batch.num_graphs, device=batch.positions.device)
        max_dist.scatter_reduce_(0, idx, dist, reduce="amax")

        keep = max_dist <= self.max_spread
        ctx.accepted_mask = keep
        if not keep.any():
            # Batch does not support zero-graph selections; signal total
            # rejection with an explicitly empty batch instead
            ctx.batch = Batch.empty(
                num_systems=0,
                num_nodes=0,
                num_edges=0,
                device=batch.positions.device,
            )
            return
        ctx.batch = batch[keep]
```

Attach the hook during generator construction:

```python
gen = AtomisticGenerator(
    generator_func=random_cluster_generate,
    hooks=[CentroidSpreadFilter(max_spread=2.5)],
)
```

## The model side: mixin and config

When generation relies on a neural network, the model subclasses
{class}`~nvalchemi.models.gen.base.GenerativeModelMixin` and declares its capabilities
via {class}`~nvalchemi.models.gen.base.GenerativeModelConfig`.

This is the non-energy counterpart to {class}`~nvalchemi.models.base.BaseModelMixin`. It
does not require energy, forces, or neighbor lists.

Every subclass must set `self.model_config` in `__init__`:

```python
import torch
from torch import nn
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.gen import GenerativeModelConfig, GenerativeModelMixin

class ToyDecoder(nn.Module, GenerativeModelMixin):
    def __init__(self, num_atoms: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_atoms = num_atoms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_atoms * 3),
        )
        self.model_config = GenerativeModelConfig(
            required_inputs=frozenset(),  # unconditional
            outputs=frozenset({"positions", "atomic_numbers"}),
            prediction_outputs={"positions"},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### Model-owning samplers

In `nvalchemi`, the model does not own the sampler. Instead, a sampler callable owns the
model:

```python
class ToyGenerate:
    def __init__(self, model: ToyDecoder) -> None:
        self.model = model
        self.required_inputs = model.model_config.required_inputs
        self.outputs = model.model_config.outputs

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def __call__(
        self,
        inputs=None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> Batch:
        z = torch.randn(
            num_samples, self.model.latent_dim, generator=rng, device=self.device
        )
        positions = self.model(z).reshape(num_samples, self.model.num_atoms, 3)
        numbers = torch.full(
            (self.model.num_atoms,), 6, dtype=torch.long, device=self.device
        )
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )
```

The driver reads `required_inputs`, `outputs`, and `device` off the sampler object as
default declarations.

For testing and rapid prototyping, `nvalchemi` provides ready-made demo models:
{class}`~nvalchemi.models.gen.demo.DemoGANModel` and
{class}`~nvalchemi.models.gen.demo.DemoDiffusionModel`.

(chaining-generators)=

## Composing pipelines with GenerationPipeline

You can chain multiple generative stages, custom transformations, and {doc}`dynamics
simulations <dynamics>` using the `|` operator:

```python
from nvalchemi.dynamics import FIRE2, ConvergenceHook
from nvalchemi.gen import AtomisticGenerator
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# Stage 1: Generate initial candidate structures
gen_stage = AtomisticGenerator(generator_func=ToyGenerate(ToyDecoder(num_atoms=8)))

# Stage 2: Relax candidates with an ML potential optimizer
relax_stage = FIRE2(
    model=DemoModelWrapper(DemoModel()),  # swap in your own model here
    dt=0.1,
    n_steps=200,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
)

# Chain into a sequential pipeline; per-stage options go through stage_kwargs
pipeline = gen_stage | relax_stage
with pipeline:  # session: shared stream, seeded RNG, stage sessions
    relaxed_batch = pipeline(stage_kwargs=[{"num_samples": 8}, None])
```

### Pipeline execution rules

{class}`~nvalchemi.gen.pipeline.GenerationPipeline` executes stages sequentially:

- **Stage invocation**: Stages with a `run()` method (such as dynamics engines) run to
  completion via `stage.run(batch, **kwargs)`. Other callables run via `stage(batch,
  **kwargs)`.
- **Field contract validation**: At construction, adjacent `AtomisticGenerator` stages
  are validated: `downstream.required_inputs` must be a subset of `upstream.outputs`.
- **Shared CUDA stream**: Entering `with pipeline:` creates a single dedicated CUDA
  stream shared across all stages honoring the `_stream` convention.
- **Stage arguments**: Pass per-stage keyword arguments using `stage_kwargs` as a list
  of dictionaries aligned with pipeline stages.

```python
out = pipeline(stage_kwargs=[{"num_samples": 8}, {"n_steps": 100}])
```

## Integrations and model families

Sampling procedures differ across model families. Below are common integration patterns.

### PhysicsNeMo diffusion

`nvalchemi` integrates directly with NVIDIA PhysicsNeMo diffusion abstractions
(`physicsnemo.diffusion`). Noise schedulers, preconditioners, and ODE/SDE samplers plug
into a generating function without adapter layers:

```python
from typing import Any
import torch
from torch import nn
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.samplers import sample as pn_sample

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen import AtomisticGenerator

class PositionDenoiser(nn.Module):
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
        b = x.shape[0]
        s = sigma.reshape(b, 1).expand(b, 1)
        return self.net(torch.cat([x.reshape(b, -1), s], dim=-1)).reshape_as(x)

class EDMGenerate:
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
        self.required_inputs = frozenset()
        self.outputs = frozenset({"positions", "atomic_numbers"})

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def __call__(
        self,
        inputs: Any = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Batch:
        denoiser = self.scheduler.get_denoiser(
            x0_predictor=EDMPreconditioner(self.model)
        )
        xN = torch.randn(
            num_samples, self.model.num_atoms, 3, generator=rng, device=self.device
        )
        xN = xN * self.scheduler.sigma_max
        x0 = pn_sample(
            denoiser, xN, self.scheduler, num_steps=self.num_steps, solver="heun"
        )
        numbers = torch.full(
            (self.model.num_atoms,), 6, dtype=torch.long, device=self.device
        )
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in x0]
        )

diffusion = AtomisticGenerator(
    generator_func=EDMGenerate(PositionDenoiser(num_atoms=32), num_steps=18),
    seed=42,
)

with diffusion:
    batch = diffusion.sample(num_samples=16)
```

Deterministic solvers (like `"heun"` or `"euler"`) derive all randomness from the
initial noise tensor `xN`. Drawing `xN` with the session's `rng` ensures exact
reproducibility from `seed`.

### Generative Adversarial Networks (GAN)

GAN generation decodes a single latent noise vector in one forward pass:

```python
class GANGenerate:
    def __init__(self, model) -> None:
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def __call__(
        self,
        inputs=None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> Batch:
        z = torch.randn(
            num_samples, self.model.latent_dim, generator=rng, device=self.device
        )
        positions = self.model.decode(z).reshape(num_samples, -1, 3)
        numbers = torch.full(
            (positions.shape[1],), 6, dtype=torch.long, device=self.device
        )
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )

from nvalchemi.models.gen import DemoGANModel

# the demo model satisfies this sketch's interface (latent_dim + decode)
gan = AtomisticGenerator(generator_func=GANGenerate(DemoGANModel()))
samples = gan(num_samples=4)
```

### Variational Autoencoders (VAE)

A VAE decodes a latent draw like a GAN, but the latent can also come from the
encoder — which is how a VAE conditions on input structures:

```python
class VAEGenerate:
    def __init__(self, model) -> None:
        self.model = model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def __call__(
        self,
        inputs=None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> Batch:
        if inputs is None:
            # unconditional: sample the prior
            z = torch.randn(
                num_samples, self.model.latent_dim, generator=rng, device=self.device
            )
            n = num_samples
        else:
            # conditional: encode the inputs, then reparametrize —
            # one draw per input graph; for repeated draws per input, add a
            # condition step that tiles the batch (the driver does not tile
            # on its own)
            mu, logvar = self.model.encode(inputs)
            n = mu.shape[0]
            eps = torch.randn(n, self.model.latent_dim, generator=rng, device=mu.device)
            z = mu + eps * torch.exp(0.5 * logvar)
        positions = self.model.decode(z).reshape(n, -1, 3)
        numbers = torch.full(
            (positions.shape[1],), 6, dtype=torch.long, device=z.device
        )
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )

vae = AtomisticGenerator(generator_func=VAEGenerate(vae_model))
samples = vae(num_samples=4)
```

For a complete executable script demonstrating model-owning samplers with both GAN and
diffusion architectures, see `examples/intermediate/09_generative_samplers.py`.

## What's next

- {doc}`Dynamics <dynamics>` — Relax or run MD on generated structures.
- {doc}`Hooks <hooks>` — Learn the shared hook protocol and reporting system.
- {doc}`Training <training>` — Train or fine-tune neural network backbones.
- {doc}`AtomicData and Batch <data>` — Understand graph data representations, node and
  edge features, and batching.
- {doc}`Generative API reference </modules/gen>` — Explore the complete class and method
  reference.

## See also

- {class}`~nvalchemi.gen.generator.AtomisticGenerator`
- {class}`~nvalchemi.gen.generator.GeneratingFunction`
- {class}`~nvalchemi.gen.generator.ConditionFunction`
- {class}`~nvalchemi.gen.stages.GenerationStage`
- {class}`~nvalchemi.hooks.GenerationContext`
- {class}`~nvalchemi.gen.pipeline.GenerationPipeline`
- {class}`~nvalchemi.models.gen.base.GenerativeModelMixin`
- {class}`~nvalchemi.models.gen.base.GenerativeModelConfig`
