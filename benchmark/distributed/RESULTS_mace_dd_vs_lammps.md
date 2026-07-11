# DD MACE (cueq + torch.compile) vs LAMMPS + MACE — H100 scaling

Head-to-head of nvalchemi's domain-decomposed (halo) MACE against a
LAMMPS + cueq (ML-IAP) + Kokkos build of the **same** MACE-MP-0 medium model.
Elongated bcc-Fe, fp32, dt = 1 fs, 200 timed steps. Metric = **ms/step**
(ours = p50 of the amortized block; LAMMPS = `1000·Loop/200`, median of the
timed blocks). Peak GPU memory in GiB. DFW H100 (80 GB). Ours = compiled DD
(cueq conv fused under compile); LAMMPS = cueq ML-IAP + Kokkos.

Methodology: see `../../benchmark-spec-lammps-vs-dd.md`.

These numbers are only meaningful because two DD bugs were fixed first — without
them every N ≳ 24k either detonated (NVT energy → NaN) or dead-locked
(compiled-halo all_to_all cap desync). See commits
`agree fixed-shape caps across ranks` and
`skip the partitioned axis when wrapping positions in the DD step`.

## world = 2 (2× H100)

| N | ours ms/step | LAMMPS ms/step | ours / LAMMPS | ours GiB | LAMMPS GiB |
|-------:|----:|----:|:---:|----:|----:|
| 1,008  |  22.2 |  24.3 | 0.91× | 1.9 | 2.2 |
| 4,032  |  27.7 |  29.8 | 0.93× | 5.4 | 6.2 |
| 7,992  |  41.6 |  38.9 | 1.07× | 10.0 | 11.2 |
| 15,984 |  70.2 |  67.5 | 1.04× | 19.3 | 21.6 |
| ~24k   |  99.7 (23976) | ~96 (interp 22032→28008) | ~1.04× | 28.5 | ~33 |
| ~32k   | 130.2 (31968) | ~123 (interp 28008→33984) | ~1.06× | 37.8 | ~42 |
| 40,032 | 159.9 | 152.4 | 1.05× | 47.2 | 52.7 |

LAMMPS w2 ceiling ≈ 64k (OOM at 69,984, 80.7 GiB). Ours uses less memory at
every N, so headroom is larger.

## world = 8 (8× H100)

| N | ours ms/step | LAMMPS ms/step | ours / LAMMPS | ours GiB | LAMMPS GiB |
|--------:|----:|----:|:---:|----:|----:|
| 7,992   |  25.3 |  29.0 | **0.87×** | 3.0 | 3.5 |
| 15,984  |  30.5 |  30.6 | **1.00×** | 5.4 | 6.2 |
| 40,032  |  51.3 |  48.1 | 1.07× | 12.3 | 13.9 |
| 64,008  |  74.0 |  68.5 | 1.08× | 19.4 | 21.6 |
| 112,032 | 118.5 | 109.5 | 1.08× | 33.3 | 37.1 |
| 159,984 | 165.7 | 151.2 | 1.10× | 47.2 | 52.6 |
| 208,008 | 203.5 | 194.3 | 1.05× | 61.1 | 68.1 |
| 256,032 | **248.7** | — (OOM) | — | 75.1 | — |

LAMMPS w8 topped out at 208k; ours reaches **256k** on the same 8 GPUs.

## Takeaways

- **Speed:** within single-digit percent of a heavily-optimized
  LAMMPS + cueq + Kokkos build across the whole range; **faster at small N**
  (w8 @8k = 0.87×), tied at 16k, ~5–10% slower at large N.
- **Memory / capacity:** ours uses **~10 % less memory at every size**, so it
  reaches larger systems before OOM (w8: 256k vs LAMMPS 208k; w2: headroom past
  the LAMMPS ~64k ceiling). The compiled cueq conv-fusion is the lever.
- **Scaling:** both near-linear in N; ours is now stable and deadlock-free at
  large N and high world size.
- The residual ~5–10 % large-N gap is dominated by the halo `all_to_all`
  (O(world) participation). Replacing it with targeted P2P scatters to
  geometric face-neighbors is the next lever (grows with world size / multinode).
