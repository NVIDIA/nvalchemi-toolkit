#!/usr/bin/env bash
# Re-runnable import-prefix rewrite for the vendored physicsnemo domain_parallel copy.
#
# WHY (correctness, not cosmetics): the upstream files contain absolute
# `from physicsnemo.domain_parallel...` imports. Left as-is inside this vendored
# copy they would resolve against the *released* physicsnemo wheel (DTensor-based
# 2.0.0), silently mixing two ShardTensor implementations. This script repoints
# them at THIS copy via relative imports. External deps
# (physicsnemo.distributed.* / .nn.* / .core.* / .utils.profiling) are stable and
# intentionally left pointing at the released wheel.
#
# RE-SYNC RECIPE (see README.md): re-copy the kept-closure files from the upstream
# branch, run ./resync.sh, re-apply the two VENDOR-EDIT manifest trims
# (domain_parallel/__init__.py cuda gate, shard_utils/__init__.py register list),
# then `git diff` against the previous vendored copy.
set -euo pipefail
DP="$(cd "$(dirname "$0")" && pwd)/domain_parallel"

# Top-level modules — package is `domain_parallel`, so the relative prefix is one dot.
for f in "$DP"/*.py; do
  sed -i -E \
    -e 's/^([[:space:]]*)from physicsnemo\.domain_parallel\./\1from ./' \
    -e 's/^([[:space:]]*)from physicsnemo\.domain_parallel import/\1from . import/' \
    -e 's/^([[:space:]]*)import physicsnemo\.domain_parallel\.shard_tensor as shard_tensor/\1from . import shard_tensor/' \
    "$f"
done

# Subpackage modules (custom_ops/, shard_utils/) — one level deeper, so two dots.
for f in "$DP"/custom_ops/*.py "$DP"/shard_utils/*.py; do
  sed -i -E \
    -e 's/^([[:space:]]*)from physicsnemo\.domain_parallel\./\1from ../' \
    -e 's/^([[:space:]]*)from physicsnemo\.domain_parallel import/\1from .. import/' \
    "$f"
done

echo "resync: rewrote intra-package imports under $DP"
