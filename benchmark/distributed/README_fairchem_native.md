# fairchem-native distributed NVT benchmark

`benchmark_fairchem_native_nvt.py` measures **fairchem's OWN graph-parallel
MD** throughput and per-rank peak memory on the exact bcc-Fe systems and
metrics used in `DD_SCALING_SWEEP_2026-07-01.md` §7a. It is the head-to-head
baseline for the toolkit's `UMA graph_partition` NVT numbers: same model,
same systems, same p50 step_ms + per-rank peak MiB — but driven entirely by
fairchem (Ray graph-parallel + ASE Langevin), no toolkit DD wrapper in the
loop.

## Why

§7a reports the toolkit's `DistributedModel` graph_partition strategy. To
claim a win (or parity) we need fairchem's native equivalent under identical
conditions. `workers=N` in `pretrained_mlip.get_predict_unit` is exactly that:
a single-system node split (`tensor_split(arange(n_atoms), N)`) across N Ray
GPU-actors — the same partition our graph_partition uses.

## The one supported path

- **Distributed GP inference is Ray-backed, NOT torchrun.**
  `get_predict_unit("uma-s-1p1", inference_settings="turbo", device="cuda",
  workers=N)` with `N > 1` returns a `ParallelMLIPPredictUnit` (one Ray
  GPU-actor per rank). Launch with `WORKERS=N python <script>` — Ray places
  the actors. **Do not wrap it in `torchrun`.**
- `inference_settings="turbo"` = compile + merge_mole + tf32, matching §7a.
- **MD driver is ASE Langevin** via `FAIRChemCalculator(predictor,
  task_name="omat")`. One `get_forces` fans out across all GP ranks; only
  rank-0 (the in-process driver worker) is visible to the MD loop.

## Environment

The UMA env only (`.tlkit-uma` / `.venv-uma`): torch 2.8, fairchem 2.21,
Ray. Needs `HF_TOKEN` for the gated UMA checkpoint. Single node only.

- **A6000 box (48 GB): worlds 1/2.** UMA turbo is ~13.7 MiB/atom, so ~3k
  atoms/GPU — keep the ladder small.
- **DFW H100 (80 GB): worlds 4/8** for the full §7a ladder.

## Peak-memory measurement (the one real obstacle)

The model runs in Ray worker processes, so the driver's
`torch.cuda.max_memory_allocated` only sees rank-0. Pick with `--mem-method`:

1. **`patch` (default, matches §7a)** — apply the small patch to the fairchem
   checkout first. It adds `reset_peak_mem` / `get_peak_mem` remote methods to
   the worker; the script resets after warmup and `ray.get`s every rank's
   `max_memory_allocated` at the end. Rank-0 is read locally (it is the
   driver); ranks 1..N-1 come from `predictor.workers`.

   ```bash
   cd fairchem
   git apply ../benchmark/distributed/fairchem_worker_peakmem.patch
   # ... run the benchmark ...
   git apply -R ../benchmark/distributed/fairchem_worker_peakmem.patch   # revert
   ```

   The patch also puts the two methods in a threaded Ray concurrency group so
   they don't deadlock behind the never-returning `predict` loop that turbo's
   `merge_mole` path parks non-rank-0 actors in.

2. **`smi` (fallback, no patch)** — samples `nvidia-smi memory.used` during
   steady state. **Caveat:** `memory.used` includes the CUDA context (hundreds
   of MiB) and is NOT directly comparable to §7a's `max_memory_allocated`.
   Footnoted as such in the output. Use only when the patch isn't applied.

## Launch commands

Box (worlds 1/2, small ladder — 48 GB ceiling):

```bash
cd fairchem && git apply ../benchmark/distributed/fairchem_worker_peakmem.patch && cd ..

WORKERS=1 /path/to/.tlkit-uma/bin/python \
    benchmark/distributed/benchmark_fairchem_native_nvt.py \
    --sizes 2000 4394 --nsteps 20 --csv fc_native_nvt_w1.csv

WORKERS=2 /path/to/.tlkit-uma/bin/python \
    benchmark/distributed/benchmark_fairchem_native_nvt.py \
    --sizes 2000 4394 8192 --nsteps 20 --csv fc_native_nvt_w2.csv
```

DFW H100 (worlds 4/8, full ladder — 80 GB):

```bash
WORKERS=4 /path/to/.tlkit-uma/bin/python \
    benchmark/distributed/benchmark_fairchem_native_nvt.py \
    --sizes 2000 4394 8192 16000 --nsteps 20 --csv fc_native_nvt_w4.csv

WORKERS=8 /path/to/.tlkit-uma/bin/python \
    benchmark/distributed/benchmark_fairchem_native_nvt.py \
    --sizes 2000 4394 8192 16000 24334 31250 --nsteps 20 --csv fc_native_nvt_w8.csv
```

Flags: `--sizes` (target atom counts; snapped to the nearest bcc `2·r³` —
the §7a ladder lands exactly), `--nsteps` (bumped automatically to at least
`--warmup` + `--timed`), `--warmup` (default 5, absorbs the first compiled
step), `--timed` (default 10, the p50 window), `--model`, `--sett` (default
`turbo`), `--mem-method {patch,smi}`, `--csv`. `WORKERS` is read from the
environment (Ray world size).

## Output

A CSV with the same columns as §7a's NVT CSV, strategy labeled
`fairchem_native_gp`:

```
model,env_torch,strategy,world,n_atoms,step_ms_p50,peak_MiB_per_rank,status
```

plus a human table on stdout. Per-size OOM is caught, recorded as
`status=OOM`, and the sweep continues to remaining sizes.

## Folding results back into §7a

The atom ladder and metrics are identical, so results drop straight into
§7a's cross-tabs as a fairchem-native comparison column:

1. Run all four worlds (box: w1/w2; DFW: w4/w8) and concatenate the CSVs.
2. For §7a-i (step_ms vs world at fixed N) and §7a-ii (peak_MiB vs world),
   add a `fairchem_native_gp` row/column beside `graph_partition` at each
   `(n_atoms, world)`.
3. Under `--mem-method patch` the peak is the same statistic as §7a
   (`max_memory_allocated`) — directly comparable. Under `--mem-method smi`,
   annotate the memory column as `nvidia-smi memory.used` (includes CUDA
   context) so it is not confused with the allocated peak.
