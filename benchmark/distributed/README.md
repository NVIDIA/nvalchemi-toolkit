# Distributed benchmarks

Performance + correctness benchmarks for domain-decomposition (DD)
inference and molecular dynamics. Two config-driven runners cover every
model; a model is selected with `--config <model>.yaml`.

| Runner | What it measures |
| --- | --- |
| `benchmark_dd_model_forward.py` | Single-GPU vs multi-GPU forward time, peak memory, and a single-vs-multi **force-equivalence** gate. |
| `benchmark_dd_nvt.py` | End-to-end NVT (`NVTLangevin`) step time across `world ∈ {0, 1, 2}`. |

Shared timing, system builders, force-gathering, and the sweep drivers
live in `_benchmark_common.py`.

## Configs

`configs/<model>.yaml` declares the test system, the model loader, and
the per-mode distribution knobs. Shipped configs:

| Config | System | Forward | NVT |
| --- | --- | :---: | :---: |
| `lj.yaml` | Argon cluster (non-PBC) | ✓ | |
| `ewald.yaml` | NaCl-like lattice (PBC, charged) | ✓ | |
| `pme.yaml` | NaCl-like lattice (PBC, charged) | ✓ | |
| `mace.yaml` | α-quartz SiO₂ supercell | ✓ | ✓ |
| `aimnet2.yaml` | Methane (CH₄) packing | ✓ | ✓ |
| `uma.yaml` | bcc iron supercell | ✓ | ✓ |

Override any config value for a one-off run with `--set` (dotted keys),
no new file needed:

```bash
--set loader.enable_cueq=true --set loader.compile=true   # MACE cueq + compile
--set loader.inference=compile                            # UMA compiled inference
--set dtype=fp32                                          # MACE same-precision (no cueq)
```

## Running

Single-GPU baseline (force-equivalence still runs single-rank only):

```bash
python benchmark/distributed/benchmark_dd_model_forward.py \
    --config benchmark/distributed/configs/lj.yaml --sizes 1000 4000 --single-only
```

Multi-GPU (force-equivalence gate active) — launch with `torchrun`:

```bash
torchrun --nproc_per_node=2 \
    benchmark/distributed/benchmark_dd_model_forward.py \
    --config benchmark/distributed/configs/mace.yaml --sizes 1000 4000
```

NVT end-to-end — run each `world` mode as a separate job (keeps allocator
pools clean between modes):

```bash
# world=0 (raw integrator)         python ... benchmark_dd_nvt.py --config ...
# world=1 (DD wrapper, single rank) torchrun --nproc_per_node=1 ... benchmark_dd_nvt.py --config ...
# world=2 (full DD)                 torchrun --nproc_per_node=2 ... benchmark_dd_nvt.py --config ...
python benchmark/distributed/benchmark_dd_nvt.py \
    --config benchmark/distributed/configs/aimnet2.yaml --sizes 500 2000
```

Without `torchrun` the forward runner reports the single-rank baseline
and the NVT runner reports `world=0`. Omit `--sizes` to use the config's
`default_sizes`. `--help` lists the shared flags (`--iters`, `--warmup`,
`--tolerance`, `--profile`, ...).

### Notes

- **MACE + cueq on multiple ranks** needs
  `CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0` to avoid a cross-rank JIT race.
- **UMA** ships in its own extras group (`uv sync --extra uma`) because
  `fairchem-core` pins a newer `e3nn` than the MACE ecosystem; it also
  needs `HF_TOKEN` for the gated checkpoints. Keep UMA and MACE in
  separate environments.
- cueq (MACE) and the AIMNet2 warp kernels are float32-only.
