# NEB benchmarks

The benchmark directory contains two independent checks:

1. `benchmark_muller_brown.py` validates NEB against a known transition state
   on the analytic Muller--Brown potential.
2. `benchmark_batched_neb.py` compares two atomistic paths run serially with
   the same paths run together as a heterogeneous batch.

## Muller--Brown correctness benchmark

The seven-image Muller--Brown case runs regular NEB followed by climbing-image
NEB. It checks the converged climbing image against the known transition-state
position `(-0.82200156, 0.62431280)` and energy `-40.6648435087` in reduced
units.

The model, endpoints, and expected transition state are defined in
`_mueller_brown_model.py`.

```bash
uv run --extra cu13 python benchmark/neb/benchmark_muller_brown.py \
  --device cuda
```

## Batched versus serial benchmark

The atomistic comparison uses two paths evaluated by the same MACE-MP
`small-0b` model:

| Path | Atoms | Periodicity | Constraint |
| --- | ---: | --- | --- |
| Cu vacancy | 31 | three-dimensional | none |
| Al(100)/Au | 13 | surface slab | bottom two layers fixed |

It first runs each path separately, then runs both paths together in one
heterogeneous batch. The script compares final energies and positions and
reports the batched speedup over the summed serial runtime.

The paired endpoint cases and their metadata are stored in `cases/`.

Run the five-image benchmark on GPU

```bash
uv run --extra cu12 --extra mace python benchmark/neb/benchmark_batched_neb.py \
  --images 5 --device cuda
```

Use `--images 9` for denser atomistic bands.

For a quick smoke test:

```bash
uv run --extra cu12 --extra mace python benchmark/neb/benchmark_batched_neb.py \
  --images 5 --device cuda \
  --max-regular-steps 2 --max-climbing-steps 2 --max-steps 6 --fmax 1e6
```

For CUDA 13, replace `cu12` with `cu13`.

## Output

The atomistic benchmark reports serial and batched timings, verifies that their
final energies and positions agree, and records the measured speedup. Expect a
speedup between 1x and 2x, depending on the hardware and workload.

The default comparison tolerances are `1e-5`; use `--energy-atol` and
`--position-atol` to change them. One warmup pass runs by default and can be
disabled with `--no-warmup`.
