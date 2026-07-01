# Vendored physicsnemo `domain_parallel` (ShardTensor compile backend)

**This is a temporary internal copy. Do not edit the `domain_parallel/`
files by hand except the two `VENDOR-EDIT` manifests below.**

## Why this exists

`torch.compile` for distributed MLIP inference requires the `torch.Tensor`-based
`ShardTensor` refactor + compile enablement from two physicsnemo PRs that are
**open, unmerged, and deferred past the 26.05 release** on an FSDP1/StormScope
*training* blocker that does not affect inference:

- **#1556** "ShardTensor Refactor" — branch `sharded_view_backwards`, head `986ac94`
- **#1682** "Enable Compile for ShardTensor" — branch `shard_tensor_compile`, head `15afcc0`
  (stacked on #1556; currently includes its diff)

Pinning to a moving branch was rejected; we vendor instead. See
`proposal-distributed-compile-vendoring.md` at the repo root for the full plan.

## Source

- Repo: `coreyjadams/physicsnemo`
- Branch: `shard_tensor_compile`
- Commit: `15afcc01d776ae0a01c6592dd984b150dee3acb3`
- Vendored on: 2026-06-01

## What was copied (file-granular verbatim)

Only the MLIP-needed dependency closure of `physicsnemo/domain_parallel/`:

```
shard_tensor.py · _shard_tensor_spec.py · _shard_redistribute.py
custom_ops/{__init__,_reductions,_tensor_ops}.py
shard_utils/{__init__,patch_core,halo,index_ops,view_ops,normalization_patches,unary_ops}.py
```

**Dropped** (grid/CFD; nothing in the closure imports them):
`shard_utils/{attention_patches, conv_patches, knn, point_cloud_ops,
natten_patches, padding, pooling_patches, unpooling_patches, mesh_ops, ring}`.

External deps (`physicsnemo.distributed.*`, `.nn.*`, `.core.version_check`,
`.utils.profiling`) are **not** vendored — they resolve against the released
`nvidia-physicsnemo` wheel, which is stable across these PRs.

## Hand-edits (the only non-verbatim changes)

1. **Import rewrite** — `resync.sh` repoints absolute
   `from physicsnemo.domain_parallel...` imports to relative so they resolve
   against this copy, not the released wheel. Deterministic and re-runnable.
2. **`VENDOR-EDIT` markers** (grep for them):
   - `domain_parallel/__init__.py` — removes the `torch.cuda.is_available()`
     gate so CPU paths register the (cuda-free) MLIP wrappers.
   - `shard_utils/__init__.py` — `register_shard_wrappers()` trimmed to the
     four MLIP wrappers (index/normalization/unary/view).

## Re-sync recipe

```bash
SHA=<new-branch-sha>
BASE="https://raw.githubusercontent.com/coreyjadams/physicsnemo/$SHA/physicsnemo/domain_parallel"
# re-fetch the kept-closure files (overwrites verbatim copies), then:
./resync.sh                       # reapply import rewrite
# reapply the two VENDOR-EDIT trims (see grep VENDOR-EDIT)
git diff                          # review what changed upstream
```

## Retirement

When physicsnemo releases the merged #1556+#1682:

1. Edit `nvalchemi/distributed/_core/_st_backend.py` to import from
   `physicsnemo.domain_parallel` instead of this package.
2. `rm -rf` this directory.
3. Run the verification ladder (proposal §8).
