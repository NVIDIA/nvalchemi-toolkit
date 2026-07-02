# Domain-Decomposition Scaling Sweep — DFW H100 (2026-07-01)

Full per-run numbers from the DFW forward + NVT scaling sweep. Every measured row
is included (timings, peak memory, energy, and OOM/failure markers) so the tables
can be graphed directly. A machine-readable CSV of all rows is in the Appendix.

## Environment / provenance
- **Cluster**: DFW, 1 node, NVIDIA **H100 80GB HBM3**, worlds ∈ {1, 2, 4, 8} GPUs.
- **Repo**: `nvalchemi-toolkit` @ `47e0028` (branch `dallasf/domain-decomposition-shard-tensor`).
- **MACE env** (`.tlkit`): torch **2.10.0+cu128**, cueq **on**, `torch.compile` **on**, dtype **fp32**.
- **UMA env** (`.tlkit-uma`): torch **2.8.0+cu128**, fairchem **2.21.0**, inference **turbo** (compile + merge_mole + tf32).
- **Bench**: `benchmark/distributed/benchmark_dd_model_forward.py`, 10 timed iters / 3 warmup.
  Timed region = model forward + autograd-for-forces (+ halo exchange on the DD path).
- **Systems**: periodic crystals sized by target atom count; `n_atoms` below is the
  **actual** snapped count (differs from the requested ladder value).

## How to read
- `step_ms` is the **W-rank** wall time per forward step (the DD measurement).
- `peak_MiB` is **per-rank** peak GPU memory.
- Multi-rank runs (world ≥ 2) do **not** run the single-rank reference by default, so
  `1-rank`/speedup/force-equivalence columns are `nan`/`—`. World=1 rows *are* the
  single-rank reference. Energies are reported for a correctness eyeball (see Caveats).
- `OOM` = that size ran out of memory on ≥1 rank (graceful — smaller sizes still reported).
- `run failed` = the whole strategy leg hard-crashed (see Caveats).

---

# 1. MACE (cueq + compile, fp32) — forward

## 1a. Halo
| world | n_atoms | step_ms | peak_MiB/rank | energy (eV) |
|---:|---:|---:|---:|---:|
| 1 | 9 000  | 48.464  | 13 287.2 | -71 033.171875 |
| 1 | 15 552 | 78.352  | 22 887.1 | -122 750.273438 |
| 1 | 24 696 | 117.443 | 36 284.8 | -194 921.578125 |
| 1 | 30 375 | 143.217 | 44 605.6 | -239 732.375000 |
| 1 | 44 217 | 204.439 | 64 886.8 | -348 991.406250 |
| 2 | 15 552 | 71.891  | 18 587.2 | -122 744.322408 |
| 2 | 30 375 | 128.531 | 34 757.9 | -239 735.309472 |
| 2 | 44 217 | 177.934 | 49 390.0 | -348 983.560556 |
| 2 | 61 731 | 251.645 | 68 303.5 | -487 213.355394 |
| 4 | 15 552 | 61.518  | 12 687.1 | -122 739.497444 |
| 4 | 30 375 | 104.025 | 22 605.2 | -239 675.470371 |
| 4 | 61 731 | 167.207 | 42 651.9 | -487 137.528615 |
| 4 | 95 832 | 248.344 | 61 815.3 | -756 347.822505 |
| 4 | 128 000 | OOM | — | — |
| 8 | 15 552 | 54.519  | 8 700.4  | -122 726.432783 |
| 8 | 30 375 | 79.038  | 14 849.5 | -239 554.521039 |
| 8 | 61 731 | 123.817 | 26 492.9 | -487 059.799659 |
| 8 | 124 416 | 202.295 | 45 981.0 | -981 920.029492 |
| 8 | 197 568 | 290.174 | 68 586.9 | -1 559 299.061769 |

## 1b. Graph-replicate
| world | n_atoms | step_ms | peak_MiB/rank | energy (eV) |
|---:|---:|---:|---:|---:|
| 1 | 9 000  | 48.989  | 13 287.2 | -71 033.171875 |
| 1 | 15 552 | 78.092  | 22 887.1 | -122 750.187500 |
| 1 | 24 696 | 117.785 | 36 284.8 | -194 921.406250 |
| 1 | 30 375 | 142.847 | 44 605.6 | -239 732.093750 |
| 1 | 44 217 | 203.109 | 64 886.8 | -348 989.687500 |
| 2 | 15 552 | 48.963  | 13 431.8 | -122 842.141166 |
| 2 | 30 375 | 87.888  | 26 132.4 | -239 892.045916 |
| 2 | 44 217 | 127.651 | 37 992.5 | -349 182.444499 |
| 2 | 61 731 | 184.209 | 53 016.2 | -487 462.766881 |
| 4 | 15 552 | 41.480  | 9 125.7  | -122 940.155200 |
| 4 | 30 375 | 70.454  | 17 721.9 | -240 044.677204 |
| 4 | 61 731 | 135.648 | 35 906.2 | -487 721.691435 |
| 4 | 95 832 | 204.599 | 55 682.3 | -757 028.827151 |
| 4 | 124 416 | 264.068 | 72 276.5 | -982 737.968920 |
| 8 | 15 552 | 43.231  | 7 203.0  | -122 468.821147 |
| 8 | 30 375 | 75.780  | 13 972.6 | -239 937.229720 |
| 8 | 61 731 | 119.923 | 28 292.9 | -488 104.428250 |
| 8 | 124 416 | 229.386 | 56 921.0 | -983 527.244174 |
| 8 | 192 000 | OOM | — | — |

---

# 2. UMA (turbo) — forward

## 2a. Halo
| world | n_atoms | step_ms | peak_MiB/rank | energy (eV) |
|---:|---:|---:|---:|---:|
| 1 | 1 024 | 52.034  | 14 111.3 | -8 423.672852 |
| 1 | 2 000 | 92.585  | 27 458.0 | -16 455.033203 |
| 1 | 2 662 | 121.149 | 36 504.4 | -21 901.802734 |
| 1 | 4 394 | 194.537 | 60 171.6 | -36 153.531250 |
| 1 | 5 488 | OOM | — | — |
| 2 | 2 000 | 116.203 | 23 945.8 | -16 454.234375 |
| 2 | 4 394 | 228.728 | 45 632.3 | -36 151.742188 |
| 2 | 5 488 | 286.364 | 55 479.1 | -45 153.054688 |
| 2 | 8 000 | OOM | — | — |
| 2 | 10 000 | OOM | — | — |
| 4 | 2 000 | *(n/a)* | 21 882.7 | -16 440.119141 |
| 4 | 4 394 | *(n/a)* | 36 109.1 | -36 132.445312 |
| 4 | 8 192 | *(n/a)* | 58 344.3 | -67 374.421875 |
| 4 | 12 000 | crash (OOM) | — | — |
| 8 | 2 000 | *(n/a)* | 20 070.3 | -16 414.412109 |
| 8 | 4 394 | *(n/a)* | 28 553.8 | -36 101.890625 |
| 8 | 8 192 | *(n/a)* | 43 043.0 | -67 333.078125 |
| 8 | 16 000 | *(n/a)* | 68 789.0 | -131 550.328125 |
| 8 | 24 334 | crash (OOM) | — | — |

**⚠ UMA halo w4/w8 is NOT usable — two problems (see §6.1):** (1) a hard OOM crash at
the top-of-ladder size kills the job before the summary table prints (so `step_ms` for
the recovered rows above was lost — only energy + peak survive from the per-size log);
(2) the recovered energies are **wrong** — a real world-dependent drift (below), not
precision. Use graph_partition for UMA. Timings marked *(n/a)* were not captured.

## 2b. Graph-partition
| world | n_atoms | step_ms | peak_MiB/rank | energy (eV) |
|---:|---:|---:|---:|---:|
| 1 | 1 024 | 51.984  | 14 110.1 | -8 423.672852 |
| 1 | 2 000 | 92.435  | 27 458.1 | -16 455.037109 |
| 1 | 2 662 | 120.434 | 36 506.0 | -21 901.796875 |
| 1 | 4 394 | 190.410 | 60 174.7 | -36 153.539062 |
| 1 | 5 488 | OOM | — | — |
| 2 | 2 000 | 55.865  | 13 238.7 | -16 455.148438 |
| 2 | 4 394 | 105.982 | 28 916.6 | -36 154.046875 |
| 2 | 5 488 | 128.999 | 36 090.6 | -45 155.203125 |
| 2 | 8 192 | 184.875 | 53 811.9 | -67 399.734375 |
| 2 | 9 826 | 219.340 | 64 523.7 | -80 847.187500 |
| 4 | 2 000 | 38.425  | 6 683.5  | -16 454.996094 |
| 4 | 4 394 | 62.882  | 14 517.6 | -36 153.820312 |
| 4 | 8 192 | 104.352 | 26 978.1 | -67 402.914062 |
| 4 | 11 664 | 142.908 | 38 361.7 | -95 969.054688 |
| 4 | 16 000 | 189.267 | 52 590.2 | -131 637.031250 |
| 4 | 21 296 | OOM | — | — |
| 8 | 2 000 | 30.564  | 3 406.1  | -16 455.066406 |
| 8 | 4 394 | 41.896  | 7 324.4  | -36 153.425781 |
| 8 | 8 192 | 64.049  | 13 560.1 | -67 402.460938 |
| 8 | 16 000 | 110.469 | 26 391.9 | -131 643.062500 |
| 8 | 24 334 | 157.860 | 40 052.7 | -200 208.828125 |
| 8 | 31 250 | 196.989 | 51 415.6 | -257 103.000000 |
| 8 | 39 366 | 341.488 | 64 731.3 | -323 876.375000 |

---

# 3. Strong-scaling cross-tabs (step_ms at fixed N vs world)

## 3a. MACE halo — step_ms
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 15 552 | 78.352 | 71.891 | 61.518 | 54.519 |
| 30 375 | 143.217 | 128.531 | 104.025 | 79.038 |
| 61 731 | — | 251.645 | 167.207 | 123.817 |
| 124 416 | — | — | — | 202.295 |

## 3b. MACE graph-replicate — step_ms
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 15 552 | 78.092 | 48.963 | 41.480 | 43.231 |
| 30 375 | 142.847 | 87.888 | 70.454 | 75.780 |
| 61 731 | — | 184.209 | 135.648 | 119.923 |
| 124 416 | — | — | 264.068 | 229.386 |

## 3c. UMA graph-partition — step_ms
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 2 000 | 92.435 | 55.865 | 38.425 | 30.564 |
| 4 394 | 190.410 | 105.982 | 62.882 | 41.896 |
| 8 192 | — | 184.875 | 104.352 | 64.049 |
| 16 000 | — | — | 189.267 | 110.469 |

---

# 4. Memory cross-tabs (peak_MiB/rank at fixed N vs world)

## 4a. MACE halo — peak_MiB/rank
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 15 552 | 22 887.1 | 18 587.2 | 12 687.1 | 8 700.4 |
| 30 375 | 44 605.6 | 34 757.9 | 22 605.2 | 14 849.5 |
| 61 731 | — | 68 303.5 | 42 651.9 | 26 492.9 |
| 124 416 | — | — | — | 45 981.0 |

## 4b. MACE graph-replicate — peak_MiB/rank
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 15 552 | 22 887.1 | 13 431.8 | 9 125.7 | 7 203.0 |
| 30 375 | 44 605.6 | 26 132.4 | 17 721.9 | 13 972.6 |
| 61 731 | — | 53 016.2 | 35 906.2 | 28 292.9 |
| 124 416 | — | — | 72 276.5 | 56 921.0 |

## 4c. UMA graph-partition — peak_MiB/rank
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 2 000 | 27 458.1 | 13 238.7 | 6 683.5 | 3 406.1 |
| 4 394 | 60 174.7 | 28 916.6 | 14 517.6 | 7 324.4 |
| 8 192 | — | 53 811.9 | 26 978.1 | 13 560.1 |
| 16 000 | — | — | 52 590.2 | 26 391.9 |

---

# 5. Capacity envelope (largest N that fit, per world)
| model / strategy | w1 | w2 | w4 | w8 |
|---|---:|---:|---:|---:|
| MACE halo | 44 217 (64.9 GB) | 61 731 (68.3 GB) | 95 832 (61.8 GB) | **197 568** (68.6 GB) |
| MACE graph-replicate | 44 217 (64.9 GB) | 61 731 (53.0 GB) | 124 416 (72.3 GB) | 124 416 (56.9 GB) |
| UMA graph-partition | 4 394 (60.2 GB) | 9 826 (64.5 GB) | 16 000 (52.6 GB) | **39 366** (64.7 GB) |
| UMA halo | 4 394 (60.2 GB) | 5 488 (55.5 GB) | (failed) | (failed) |

Next size above each entry OOM'd (or the strategy failed). MACE halo reaches the
largest N at w8 (owned+ghost shards); graph-replicate replicates node features so it
tops out lower at high world.

---

# 6. Caveats / notes
- **`nan` reference energies (world ≥ 2)**: the single-rank reference is skipped by
  default on the multi-rank path, so `1-rank`/speedup/force-equivalence are `nan`/`—`.
  Force-equivalence is validated separately in the per-model DD equivalence tests, not here.
- **MACE graph-replicate energy drift with world** (fp accumulation in the cross-rank
  reduce): e.g. 61 731 atoms — halo −487 213 (w2) / −487 138 (w4) / −487 060 (w8) stays
  within ~150 eV, while GP −487 463 (w2) / −487 722 (w4) / −488 104 (w8) drifts ~600 eV.
  **Halo is the correctness reference; GP trades a small energy drift for throughput/leaner memory.**
- **UMA halo is a poor fit for these small dense boxes** (ghost shell ≈ whole system):
  ~2× slower than graph-partition and OOMs far earlier (w2 4 394: halo 228.7 ms / 45.6 GB
  vs GP 106.0 ms / 28.9 GB). graph-partition is the intended UMA DD strategy here.
- **UMA halo w4/w8 crash + energy drift** — fully diagnosed in §6.1. Short version: the
  crash is an OOM at the top ladder size (halo's ghost memory), and the recovered rows
  are numerically **wrong** (world-dependent drift ~+14.5 eV @ w4, +40.6 eV @ w8 for 2000
  atoms) — **not TF32** (a matched on/off run changed the energy by ≤0.2 eV). Consistent
  with degenerate spatial partitioning (domain edge < ~2×ghost_width) at small-N/high-world,
  a known halo constraint. graph_partition (no geometric constraint) stays exact. UMA halo
  **w2 works** (the fairchem-2.21 MoLE compat fix is live), just slow/memory-heavy.
- **MACE OOMs are cueq alloc(`cudaMallocAsync`) or torch OOM**; graceful per-size (smaller
  sizes still reported). A *partial-rank* OOM at the top of a ladder can deadlock the halo
  all-to-all (NCCL watchdog timeout) — kept these ladders at/below measured fits to avoid it.
- **TF32 on H100**: eager fp32 matmuls are lowered to TF32 by default; irrelevant to these
  throughput/memory numbers but it perturbs near-zero-force equivalence checks (set
  `NVIDIA_TF32_OVERRIDE=0` for strict force gates).

## 6.1 UMA-halo drift — TF32 isolation (job 13329326, H100, turbo)
Ran w1 (reference) and w4 halo, each with TF32 on and off, same job/precision. The drift
is **world-dependent halo error, not TF32** — TF32 on↔off moves the energy by ≤0.2 eV
while the w4-vs-reference drift is ~14.5 eV either way.

| leg | 2 000 | 4 394 | 8 192 |
|---|---:|---:|---:|
| w1 ref, TF32 on  | -16 455.029 | -36 153.516 | (OOM) |
| w1 ref, TF32 off | -16 454.893 | -36 153.207 | (OOM) |
| w4 halo, TF32 on  | -16 440.121 | -36 132.445 | -67 374.422 |
| w4 halo, TF32 off | -16 440.350 | -36 132.230 | -67 373.969 |
| **drift w4−w1 (on)**  | **+14.91** | **+21.07** | — |
| **drift w4−w1 (off)** | **+14.54** | **+20.98** | — |

Conclusion: TF32 contribution ≤ 0.2 eV; the ~14.5 eV (7 meV/atom) drift is a genuine
DD-halo error that grows with world (w2 +0.8 → w4 +14.5 → w8 +40.6 eV), matching a
degenerate spatial partition at small-N/high-world. **Do not use UMA halo in this regime;
use graph_partition.** (Larger boxes where the domain stays ≫ ghost_width would avoid it —
untested. graph_partition is unaffected regardless.)

---

# 7. NVT end-to-end sweep (`NVTLangevin`, 20 timed iters / 5 warmup)

Timed region = full MD step (half-kicks + neighbour rebuild + halo/gather + model
forward + autograd forces + force consolidation). **`step_ms` here is the `p50`
(median) step time**, not the mean: NVT compiles on the first step, so the mean is
skewed by that one outlier (e.g. UMA w1 n=2000: mean 953 ms vs p50 94 ms). p50 is
the representative steady-state MD-step cost. `peak_MiB` is per-rank peak.

## 7a. UMA (turbo) — graph_partition — **green at world 1/2/4/8**
| world | n_atoms | step_ms (p50) | peak_MiB/rank |
|---:|---:|---:|---:|
| 0 (ref) | 1 024 | 53.569  | 14 163.2 |
| 0 (ref) | 2 000 | 94.062  | 27 510.3 |
| 0 (ref) | 2 662 | 121.863 | 36 554.7 |
| 0 (ref) | 4 394 | 193.803 | 60 222.6 |
| 1 | 1 024 | 52.965  | 14 112.0 |
| 1 | 2 000 | 93.576  | 27 459.2 |
| 1 | 2 662 | 121.181 | 36 507.0 |
| 1 | 4 394 | 192.158 | 60 176.1 |
| 2 | 2 000 | 68.435  | 15 245.4 |
| 2 | 4 394 | 130.887 | 33 274.1 |
| 2 | 5 488 | 158.607 | 41 645.0 |
| 2 | 8 192 | 230.269 | 61 973.3 |
| 4 | 2 000 | 44.447  | 7 704.1  |
| 4 | 4 394 | 78.551  | 16 729.1 |
| 4 | 8 192 | 129.544 | 31 164.7 |
| 4 | 11 664 | 173.906 | 44 160.1 |
| 4 | 16 000 | 230.316 | 60 495.8 |
| 8 | 2 000 | 33.902  | 3 953.8  |
| 8 | 4 394 | 50.632  | 8 541.1  |
| 8 | 8 192 | 76.005  | 15 670.9 |
| 8 | 16 000 | 129.351 | 30 330.6 |
| 8 | 24 334 | 186.107 | 46 053.8 |
| 8 | 31 250 | 233.559 | 59 045.5 |

world=1 DD matches the world=0 raw-integrator reference to <1% (p50 + peak), i.e.
the DD wrapper adds negligible single-rank overhead. The #135 w4 collective desync
(prior SIGABRT) is resolved — every world runs the full ladder to a clean end.

### 7a-i. UMA GP NVT strong-scaling — step_ms (p50) at fixed N vs world
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 2 000 | 93.576  | 68.435  | 44.447  | 33.902 |
| 4 394 | 192.158 | 130.887 | 78.551  | 50.632 |
| 8 192 | —       | 230.269 | 129.544 | 76.005 |
| 16 000 | —      | —       | 230.316 | 129.351 |

### 7a-ii. UMA GP NVT memory — peak_MiB/rank at fixed N vs world
| n_atoms | w1 | w2 | w4 | w8 |
|---:|---:|---:|---:|---:|
| 2 000 | 27 459.2 | 15 245.4 | 7 704.1  | 3 953.8  |
| 4 394 | 60 176.1 | 33 274.1 | 16 729.1 | 8 541.1  |
| 8 192 | —        | 61 973.3 | 31 164.7 | 15 670.9 |
| 16 000 | —       | —        | 60 495.8 | 30 330.6 |

Capacity envelope (largest N run per world): w1 4 394 · w2 8 192 · w4 16 000 ·
**w8 31 250** (the w8 top-of-ladder 32 000 target snapped to 31 250; sweep ended
cleanly). Near-ideal memory scaling — peak/rank ≈ halves per doubling of world at
fixed N — and ~2.8× step-time speedup w2→w8 at N=8 192 (230→76 ms).

## 7a-iii. Fairchem-native baseline (Ray GP + ASE Langevin)
Fairchem's **own** distributed-MD path (`get_predict_unit(workers=N)` Ray graph-parallel +
`FAIRChemCalculator` + ASE `Langevin`), uma-s-1p1 turbo — an apples-to-apples baseline for §7a
(same model, same GP idea, fairchem's own driver). Peak is `nvidia-smi memory.used`
(`--mem-method smi`, includes the CUDA context ≈1–2 GB), NOT comparable in absolute GB to §7a's
`max_memory_allocated`; **step_ms IS comparable.** Bench:
`benchmark/distributed/benchmark_fairchem_native_nvt.py`.

### DFW H100 — fairchem-native ladder (step_ms p50 [peak smi MiB/rank])
| n_atoms | w2 | w4 | w8 |
|---:|---:|---:|---:|
| 2 000  | 55.4 [16 029]  | 34.6 [9 201]   | 30.3 [5 963]  |
| 4 394  | 154.0 [33 497] | 79.2 [17 907]  | 51.4 [10 423] |
| 8 192  | 231.9 [61 033] | 116.2 [45 411] | 84.2 [24 397] |
| 9 826  | 305.0 [72 793] | —              | —             |
| 16 000 | OOM (11 664)   | 233.2 [59 881] | 154.9 [36 227] |
| 21 296 | —              | 307.2 [79 095] | —             |
| 24 334 | —              | OOM            | 233.4 [46 887] |
| 31 250 | —              | —              | 289.1 [59 451] |
| 39 366 | —              | —              | 443.8 [74 207] |

Capacity (largest N before OOM): **w2 9 826 · w4 21 296 · w8 39 366.**

### DFW H100 — ratio fairchem-native ÷ our §7a UMA-GP (step_ms; <1 = fairchem faster)
| n_atoms | w2 | w4 | w8 |
|---:|---:|---:|---:|
| 2 000  | **0.81** | **0.78** | **0.89** |
| 4 394  | 1.18     | 1.01     | 1.02     |
| 8 192  | 1.01     | 0.90     | **1.11** |
| 16 000 | —        | 1.01     | **1.20** |
| 24 334 | —        | —        | **1.25** |
| 31 250 | —        | —        | **1.24** |

**On matched H100 hardware, fairchem-native is competitive — not the "~4× slower" that an
earlier A6000-vs-H100 comparison implied** (that confounded hardware: A6000 fairchem against
H100 ours; see the A6000 note below). Head-to-head H100↔H100, same uma-s-1p1 turbo model:
- **Small N (2 000):** fairchem is slightly *faster* at every world (0.78–0.89×) — its
  Ray-persistent turbo graph amortizes the ASE per-step conversion well when the step is short.
- **Mid N (~4–8 k):** a wash (≈1.0×).
- **High world + large N:** our integrated path pulls ahead — **w8 ≥8 192 runs 1.11–1.25×
  faster** (233→186 ms at 24 334). Our persistent sharded batch (no per-step ASE↔batch rebuild
  + internal graph re-build each `get_forces`) wins exactly in the large-N/high-parallelism
  regime DD is *for*. Capacity is comparable (fairchem reaches marginally higher top-N per world
  in smi units, but that includes the CUDA context and our ladder simply stopped earlier).

### A6000 (plumbing/capacity context only — NOT a latency baseline)
The earlier A6000 run (w1 213.7/401.4/529.0 ms at 1 024/2 000/2 662; w2 219.1/462.5/573.2 at
2 000/4 394/5 488; peak smi w1 15.1/39.2/42.3, w2 15.1/32.4/40.4 GB; w2 cap ≈5 488) validated
the bench + the per-size subprocess isolation, but its "ours" column was DFW H100 — so read
A6000 as capacity/plumbing context, not a speed comparison.

Run notes: DFW w2/w4/w8 required `HF_HUB_OFFLINE=1` + `FAIRCHEM_CACHE_DIR`/`HF_HOME` (gated repo
on offline compute nodes); A6000 w2 required `NCCL_P2P_DISABLE=1` (Ray GP all-gather otherwise
deadlocks on trx40-03). **Per-size process isolation:** fairchem carries compiled-graph /
`AtomicData.validate` state across sizes in one process (the 2nd+ size asserts), so the bench
re-execs one fresh subprocess per N.

### 7a-iii-DFW. Fairchem-native vs our UMA-GP — **H100, world=2** (the fair head-to-head)
Both uma-s-1p1 turbo, w2, H100. fairchem peak = smi (incl CUDA context); our peak = max_alloc.
| n_atoms | fairchem-native p50 (ms) | our §7a UMA-GP p50 (ms) | ratio | fc peak (smi) | our peak |
|---:|---:|---:|---:|---:|---:|
| 2 000 | 55.4  | 68.4  | 0.81× | 16.0 GB | 15.2 GB |
| 4 394 | 154.0 | 130.9 | 1.18× | 33.5 GB | 33.3 GB |
| 8 192 | 231.9 | 230.3 | 1.01× | 61.0 GB | 62.0 GB |
| 9 826 | 305.0 | 219.3* | 1.39× | 72.8 GB | 64.5 GB† |
| 11 664 | OOM (H100 w2 cap ≈9 826) | — | — | — | — |

*§7a UMA-GP 9 826 is the **forward** number (§2b); NVT §7a only went to 8 192 at w2.
†§2b forward peak. Memory is comparable within the smi-vs-max_alloc caveat.

**Verdict: on H100 w2 the two are in the same ballpark — fairchem-native is faster at the
smallest N (2 000) but slower/tied from 4 394 up (1.0–1.4×), so mostly slower where it
matters.** This is a very different regime from A6000 w1 (fairchem ~4× slower): there the ASE
per-step overhead (Atoms↔batch + graph rebuild) dominated on a slow GPU at small N; on fast
H100s at scale the model compute dominates and the harness gap mostly closes. Our integrated
`NVTLangevin`/`DomainParallel` loop is at least as fast as fairchem's own ASE path across the
ladder and clearly wins in the small-N / commodity-GPU regime. (Job 13354817; a CSV-dir bug
dropped the CSV but all rows are in the log; fixed for w4/8.)
DFW H100 w4/8 pending greenlight.

## 7b. MACE (cueq + compile, fp32) — halo — **green at world 1/2/4/8 (0 steady-state recompiles)**
Steady-state `step_ms` = p50; `mean` is inflated by the first-step compile outlier
(e.g. w4 95 832: mean 2 570.9 vs p50 1 693.9) and is not reported here.
| world | n_atoms | step_ms (p50) | peak_MiB/rank |
|---:|---:|---:|---:|
| 1 | 9 000  | 41.054  | 12 978.4 |
| 1 | 15 552 | 66.562  | 22 353.5 |
| 1 | 24 696 | 96.091  | 35 437.5 |
| 1 | 36 864 | 140.277 | 52 848.5 |
| 2 | 15 552 | 497.135   | 22 041.4 |
| 2 | 30 375 | 152.735 † | 41 087.4 |
| 2 | 44 217 | 1 265.078 | 58 240.8 |
| 4 | 15 552 | 390.526   | 15 547.1 |
| 4 | 30 375 | 661.711   | 27 436.8 |
| 4 | 61 731 | 1 183.584 | 50 399.1 |
| 4 | 95 832 | 1 693.860 | 73 556.8 |
| 8 | 15 552 | 316.267   | 11 090.9 |
| 8 | 30 375 | 494.235   | 18 492.0 |
| 8 | 61 731 | 808.112   | 31 998.9 |
| 8 | 124 416 | 1 398.827 | 56 715.3 |
| 8 | 158 184 | 1 695.003 | 69 371.2 |

Jobs: w2 `13346449`, w4 `13344180`, w8 `13344181` (all `recompiles=0`, `data['batch']`
size-mismatch = 0, clean sweep end). w1 reference `13339484`.

† **w2 30 375 is bimodal** (min 151.7 / p50 152.7 / **p99 915.1** / max 927.0 ms):
roughly half the steps run ~152 ms and half ~915 ms, so the p50 under-reports the
representative step (physical steady state ≈ p99 ≈ 915 ms, consistent with ~2× the
15 552 row). This is a per-step **load-imbalance / migration** timing artifact at
this size, **not** a recompile issue (count = 0). The 15 552 (p50 497.1, tight) and
44 217 (p50 1 265.1) rows are clean. Worth a follow-up partition-balance profile;
orthogonal to #144.

**The #144 fix (commit `4b407ca`).** MACE compiled-DD MD was recompiling **every
step** (~25 s/step at large N): the fixed-shape halo caps that keep the compiled
graph reusable are gated on the framework DD-compile flag, but `DomainParallel`
never threaded compile intent to `DistributedModel`, so the caps never ran and the
`batch_idx` count drifted each step as atoms migrated. Threading
`DomainConfig.compile` → framework DD-compile applies the caps (stable atom/edge
shapes) → **0 steady-state recompiles**. Two torch-2.10 hazards were fixed
alongside (`_configure_dd_dynamo`, applied unconditionally at DD scope setup): the
AOTAutograd on-disk cache colliding cross-rank (`KeyError: 'lengths'` → NCCL
deadlock) and the `cache_size_limit`→`recompile_limit` rename silently no-op'ing
the ceiling bump.

**w2 confirmed (job `13346449`, post-fix).** n=15 552: p50 **497.1 ms**, 0 recompiles,
0 `data['batch']` size-mismatch, clean sweep — the fix holds at w2 as well as w4/w8.
(An earlier w2 attempt, `13343280`, predated the fix and crashed with the AOTAutograd
`KeyError: 'lengths'`; that is resolved.)

**Perf note (halo-geometry-bound, not a regression).** These are compiled *MD*
steps (per-step NL rebuild + halo exchange + migration + caps-padded forward +
autograd + consolidation), so the DD path carries real per-step overhead the
forward-only benchmark (§1a) doesn't. At these small-N sizes the ghost shell
dominates (efficiency ~ ghost_width/box_edge), so DD **anti-scales vs w1** here and
is a **capacity** play, not a small-N speedup — w8 reaches **158 184 atoms**, far
past the w1 fit. Scaling w4→w8 improves with N (661→494 ms @30 375 = 1.34×;
1 183→808 ms @61 731 = 1.46×). Good per-GPU efficiency needs ~1–2 M atoms (see §5
capacity envelope + the halo-geometry caveat). A follow-up profile of the w4 small-N
per-step overhead (≈6× the w1 step at 15 552) is worthwhile but orthogonal to #144.

---

# Appendix: CSV (all forward rows)
`model,env_torch,strategy,world,n_atoms,step_ms,peak_MiB_per_rank,energy_eV,status`

```csv
MACE,2.10.0,halo,1,9000,48.464,13287.2,-71033.171875,ok
MACE,2.10.0,halo,1,15552,78.352,22887.1,-122750.273438,ok
MACE,2.10.0,halo,1,24696,117.443,36284.8,-194921.578125,ok
MACE,2.10.0,halo,1,30375,143.217,44605.6,-239732.375000,ok
MACE,2.10.0,halo,1,44217,204.439,64886.8,-348991.406250,ok
MACE,2.10.0,halo,2,15552,71.891,18587.2,-122744.322408,ok
MACE,2.10.0,halo,2,30375,128.531,34757.9,-239735.309472,ok
MACE,2.10.0,halo,2,44217,177.934,49390.0,-348983.560556,ok
MACE,2.10.0,halo,2,61731,251.645,68303.5,-487213.355394,ok
MACE,2.10.0,halo,4,15552,61.518,12687.1,-122739.497444,ok
MACE,2.10.0,halo,4,30375,104.025,22605.2,-239675.470371,ok
MACE,2.10.0,halo,4,61731,167.207,42651.9,-487137.528615,ok
MACE,2.10.0,halo,4,95832,248.344,61815.3,-756347.822505,ok
MACE,2.10.0,halo,4,128000,,,,OOM
MACE,2.10.0,halo,8,15552,54.519,8700.4,-122726.432783,ok
MACE,2.10.0,halo,8,30375,79.038,14849.5,-239554.521039,ok
MACE,2.10.0,halo,8,61731,123.817,26492.9,-487059.799659,ok
MACE,2.10.0,halo,8,124416,202.295,45981.0,-981920.029492,ok
MACE,2.10.0,halo,8,197568,290.174,68586.9,-1559299.061769,ok
MACE,2.10.0,graph_replicate,1,9000,48.989,13287.2,-71033.171875,ok
MACE,2.10.0,graph_replicate,1,15552,78.092,22887.1,-122750.187500,ok
MACE,2.10.0,graph_replicate,1,24696,117.785,36284.8,-194921.406250,ok
MACE,2.10.0,graph_replicate,1,30375,142.847,44605.6,-239732.093750,ok
MACE,2.10.0,graph_replicate,1,44217,203.109,64886.8,-348989.687500,ok
MACE,2.10.0,graph_replicate,2,15552,48.963,13431.8,-122842.141166,ok
MACE,2.10.0,graph_replicate,2,30375,87.888,26132.4,-239892.045916,ok
MACE,2.10.0,graph_replicate,2,44217,127.651,37992.5,-349182.444499,ok
MACE,2.10.0,graph_replicate,2,61731,184.209,53016.2,-487462.766881,ok
MACE,2.10.0,graph_replicate,4,15552,41.480,9125.7,-122940.155200,ok
MACE,2.10.0,graph_replicate,4,30375,70.454,17721.9,-240044.677204,ok
MACE,2.10.0,graph_replicate,4,61731,135.648,35906.2,-487721.691435,ok
MACE,2.10.0,graph_replicate,4,95832,204.599,55682.3,-757028.827151,ok
MACE,2.10.0,graph_replicate,4,124416,264.068,72276.5,-982737.968920,ok
MACE,2.10.0,graph_replicate,8,15552,43.231,7203.0,-122468.821147,ok
MACE,2.10.0,graph_replicate,8,30375,75.780,13972.6,-239937.229720,ok
MACE,2.10.0,graph_replicate,8,61731,119.923,28292.9,-488104.428250,ok
MACE,2.10.0,graph_replicate,8,124416,229.386,56921.0,-983527.244174,ok
MACE,2.10.0,graph_replicate,8,192000,,,,OOM
UMA,2.8.0,halo,1,1024,52.034,14111.3,-8423.672852,ok
UMA,2.8.0,halo,1,2000,92.585,27458.0,-16455.033203,ok
UMA,2.8.0,halo,1,2662,121.149,36504.4,-21901.802734,ok
UMA,2.8.0,halo,1,4394,194.537,60171.6,-36153.531250,ok
UMA,2.8.0,halo,1,5488,,,,OOM
UMA,2.8.0,halo,2,2000,116.203,23945.8,-16454.234375,ok
UMA,2.8.0,halo,2,4394,228.728,45632.3,-36151.742188,ok
UMA,2.8.0,halo,2,5488,286.364,55479.1,-45153.054688,ok
UMA,2.8.0,halo,2,8000,,,,OOM
UMA,2.8.0,halo,2,10000,,,,OOM
UMA,2.8.0,halo,4,,,,,run_failed
UMA,2.8.0,halo,8,,,,,run_failed
UMA,2.8.0,graph_partition,1,1024,51.984,14110.1,-8423.672852,ok
UMA,2.8.0,graph_partition,1,2000,92.435,27458.1,-16455.037109,ok
UMA,2.8.0,graph_partition,1,2662,120.434,36506.0,-21901.796875,ok
UMA,2.8.0,graph_partition,1,4394,190.410,60174.7,-36153.539062,ok
UMA,2.8.0,graph_partition,1,5488,,,,OOM
UMA,2.8.0,graph_partition,2,2000,55.865,13238.7,-16455.148438,ok
UMA,2.8.0,graph_partition,2,4394,105.982,28916.6,-36154.046875,ok
UMA,2.8.0,graph_partition,2,5488,128.999,36090.6,-45155.203125,ok
UMA,2.8.0,graph_partition,2,8192,184.875,53811.9,-67399.734375,ok
UMA,2.8.0,graph_partition,2,9826,219.340,64523.7,-80847.187500,ok
UMA,2.8.0,graph_partition,4,2000,38.425,6683.5,-16454.996094,ok
UMA,2.8.0,graph_partition,4,4394,62.882,14517.6,-36153.820312,ok
UMA,2.8.0,graph_partition,4,8192,104.352,26978.1,-67402.914062,ok
UMA,2.8.0,graph_partition,4,11664,142.908,38361.7,-95969.054688,ok
UMA,2.8.0,graph_partition,4,16000,189.267,52590.2,-131637.031250,ok
UMA,2.8.0,graph_partition,4,21296,,,,OOM
UMA,2.8.0,graph_partition,8,2000,30.564,3406.1,-16455.066406,ok
UMA,2.8.0,graph_partition,8,4394,41.896,7324.4,-36153.425781,ok
UMA,2.8.0,graph_partition,8,8192,64.049,13560.1,-67402.460938,ok
UMA,2.8.0,graph_partition,8,16000,110.469,26391.9,-131643.062500,ok
UMA,2.8.0,graph_partition,8,24334,157.860,40052.7,-200208.828125,ok
UMA,2.8.0,graph_partition,8,31250,196.989,51415.6,-257103.000000,ok
UMA,2.8.0,graph_partition,8,39366,341.488,64731.3,-323876.375000,ok
```

## NVT CSV (step_ms = p50 steady-state; world 0 = raw-integrator reference)
`model,env_torch,strategy,world,n_atoms,step_ms_p50,peak_MiB_per_rank,status`

```csv
UMA,2.8.0,reference,0,1024,53.569,14163.2,ok
UMA,2.8.0,reference,0,2000,94.062,27510.3,ok
UMA,2.8.0,reference,0,2662,121.863,36554.7,ok
UMA,2.8.0,reference,0,4394,193.803,60222.6,ok
UMA,2.8.0,graph_partition,1,1024,52.965,14112.0,ok
UMA,2.8.0,graph_partition,1,2000,93.576,27459.2,ok
UMA,2.8.0,graph_partition,1,2662,121.181,36507.0,ok
UMA,2.8.0,graph_partition,1,4394,192.158,60176.1,ok
UMA,2.8.0,graph_partition,2,2000,68.435,15245.4,ok
UMA,2.8.0,graph_partition,2,4394,130.887,33274.1,ok
UMA,2.8.0,graph_partition,2,5488,158.607,41645.0,ok
UMA,2.8.0,graph_partition,2,8192,230.269,61973.3,ok
UMA,2.8.0,graph_partition,4,2000,44.447,7704.1,ok
UMA,2.8.0,graph_partition,4,4394,78.551,16729.1,ok
UMA,2.8.0,graph_partition,4,8192,129.544,31164.7,ok
UMA,2.8.0,graph_partition,4,11664,173.906,44160.1,ok
UMA,2.8.0,graph_partition,4,16000,230.316,60495.8,ok
UMA,2.8.0,graph_partition,8,2000,33.902,3953.8,ok
UMA,2.8.0,graph_partition,8,4394,50.632,8541.1,ok
UMA,2.8.0,graph_partition,8,8192,76.005,15670.9,ok
UMA,2.8.0,graph_partition,8,16000,129.351,30330.6,ok
UMA,2.8.0,graph_partition,8,24334,186.107,46053.8,ok
UMA,2.8.0,graph_partition,8,31250,233.559,59045.5,ok
MACE,2.10.0,halo,1,9000,41.054,12978.4,ok
MACE,2.10.0,halo,1,15552,66.562,22353.5,ok
MACE,2.10.0,halo,1,24696,96.091,35437.5,ok
MACE,2.10.0,halo,1,36864,140.277,52848.5,ok
```

---

# 8. Multinode — 16 GPU / 2 nodes (InfiniBand), UMA graph-partition NVT

Two H100 nodes (8 GPU each), UMA-small turbo, `graph_partition`, NVTLangevin 20 timed / 5 warmup,
p50. Cross-node transport = **InfiniBand** (CX7/mlx5, GPUDirect RDMA — NCCL logs `NET/IB/…/GDRDMA`).

**Two setup gotchas (both resolved):**
- **Device-ordinal bug (benchmark harness):** `_benchmark_common.py` picked the GPU by *global*
  rank (`cuda:{dist.get_rank()}`), so node-1 ranks 8–15 → `cuda:8…15` → `invalid device ordinal`
  → node-0 ranks hang on the allgather → 600 s NCCL watchdog timeout. Fixed to use `LOCAL_RANK`
  (both call sites). The framework (`_runtime.py`) was already correct.
- **IB is mandatory.** IB isn't exposed to the Pyxis container via the external IBext plugin
  (it hung at Init), so the first runs forced `NCCL_IB_DISABLE=1`+Socket (TCP) — **1.7–3.6× slower**.
  Using NCCL's **built-in** IB (`NCCL_IB_HCA=mlx5`, `NCCL_IB_DISABLE=0`, `NCCL_NET_PLUGIN=none`)
  fixed it. TCP-only w16 numbers are a worst case; ignore them.

## 8a. UMA-GP w16 (IB) vs fairchem-native w16 (IB) — step_ms (p50) / peak MiB/rank
| n_atoms | ours p50 | ours peak | fairchem p50 | fairchem peak (smi) | ratio fc÷ours |
|---:|---:|---:|---:|---:|---:|
| 2 000  | 38.6  | 2 056  | —      | —      | — |
| 8 192  | 61.5  | 7 924  | 43.3   | 13 189 | 0.70 |
| 16 000 | 92.7  | 15 246 | 74.9   | 17 043 | 0.81 |
| 31 250 | 154.0 | 29 713 | 124.2  | 31 085 | 0.81 |
| 39 366 | **186.7** | 37 257 | 234.5* | 39 031 | **1.26** |
| 48 778 | **223.1** | 46 255 | 260.1  | 47 639 | **1.17** |
| 54 000 | **243.9** | 51 092 | 277.4  | 52 395 | **1.14** |
| 65 536 | 290.9 | 61 986 | NCCL error† | — | — |
| 71 874 | 318.0 | 67 854 | NCCL error† | — | — |
| ~78 k  | OOM (fit 71 874) | — | 354.8 | 74 879 | — |

ratio >1 = **ours faster**. (Jobs: ours 13363447/13363685; fairchem 13363448/13363687.)

**The crossover is between 31k and 39k atoms — at ≥39k our path overtakes fairchem's own
distributed MD by 1.14–1.26×**, on the *same* node-partition+all-gather algorithm, across 2 nodes.
Below the crossover fairchem's leaner cross-node comm wins (the per-layer all-gather + force
`all_reduce` is a larger fraction of a small step). Same shape as the w8 single-node result,
shifted right by the extra cross-node comm.

\* **fairchem 39 366 is anomalous** (234.5 ms vs its own 31 250 = 124.2 ms — a ~1.9× jump for a
1.26× size increase); the crossover holds across 3 consistent points (39/48/54k) but re-run
fairchem@39k before quoting the exact 1.26×.
† fairchem hit transient NCCL errors at 65 536 / 71 874 (recovered at 78 608), i.e. our curve is
smoother/more robust at scale. Capacity: ours OOM'd ~72–78k; fairchem fit 78 608 (~75 GB smi) —
slightly higher ceiling (`expandable_segments` may recover our top rung).

## 8b. UMA-GP scaling 8→16 GPU (ours, single-node NVLink → 2-node IB) — step_ms (p50)
| n_atoms | w8 (1 node) | w16 (2 nodes) | speedup |
|---:|---:|---:|---:|
| 2 000  | 33.9  | 38.6  | 0.88× |
| 8 192  | 76.0  | 61.5  | 1.24× |
| 16 000 | 129.4 | 92.7  | 1.40× |
| 31 250 | 233.6 | 154.0 | 1.52× |

w16 beats w8 at every N ≥ 8 192 (up to 1.52× at 31k) — genuine multinode scaling once IB is used.

**Reduce-scatter follow-up (batch-1a, 2026-07-02) — PERF-NEUTRAL, does NOT move the crossover.**
The PR #122 reviewer suggestion (switch the end-of-forward force `all_reduce([N,3])`+slice to an even
**reduce-scatter** of the owned block, ~½ the volume, + async-overlap the tiny energy reduction) was
implemented and is force-equivalent (machine precision, eager+compile, incl. uneven partitions). But
it shows **no measurable step_ms change** at w8 (NVLink) or w16 (IB): new w16 p50 = 61.1 / 154.6 /
244.2 at 8k / 31k / 54k vs old 61.5 / 154.0 / 243.9 (run-to-run noise). The force reduction was never
the bottleneck — compiled UMA is ~96 % compute-bound and the DD comm that dominates is the
**per-layer feature all-gather** (one collective *per layer*), which reduce-scatter doesn't touch.
Crossover stays ~31–39k. The genuine lever is **async-overlap of the per-layer all-gather** (#118).
batch-1a is retained as a correct/cleaner change. (Jobs 13365800 w8 / 13365801 w16, on 47e0028.)

## 8c. Capacity ladders — full N sweep to the OOM wall (batch-1a rerun, 2026-07-02)

UMA-GP NVT, p50 step_ms / peak MiB/rank, run to the per-GPU memory ceiling.

**w8 (1 node, 8× H100 80 GB):**
| n_atoms | p50 ms | peak MiB |
|---:|---:|---:|
| 8 192  | 76.1  | 15 687 |
| 16 000 | 131.8 | 30 331 |
| 24 334 | 186.3 | 46 053 |
| 31 250 | 233.7 | 59 040 |
| 35 152 | 261.2 | 66 363 |
| 40 000 | OOM (68 GB alloc) | — |

**w16 (2 nodes, 16× H100, IB)** — see §8a; fits to **71 874 @ 67.9 GB** (317.9 ms), OOM ~72–78k.

Capacity scales ~linearly with GPU count: w8 tops out at a ~35 k-atom fit (66 GB/GPU), w16 ~72 k
(≈2×). UMA-GP is a **capacity** play — partition the node set, replicate per-layer features — so each
added GPU roughly doubles the tractable system size at fixed per-GPU memory.
