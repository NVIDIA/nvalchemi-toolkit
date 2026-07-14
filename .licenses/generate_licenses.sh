#!/usr/bin/env bash
#
# generate_licenses.sh — build a throwaway venv containing EVERY optional
# dependency + dependency group, then run pip-licenses to regenerate the
# .licenses/ artifacts for open-source legal review.
#
# The extras conflict by design (cu12 vs cu13, mace's e3nn==0.4.4 vs uma's
# e3nn>=0.5, uma's torch<2.9 vs the cuXX torch>=2.11). We do NOT need a working
# environment — only complete package+license metadata. `uv pip install` does
# not prune, so installing each conflict-free group in its own pass accumulates
# the UNION of all package names. Same-named conflicting packages settle on the
# last-installed version, which is irrelevant for a license inventory.
#
# Usage:
#   .licenses/generate_licenses.sh
#
# Env overrides:
#   PYTHON=3.12            python version for the venv (default 3.11)
#   ENV_DIR=.venv-licenses location of the throwaway venv
#   REUSE_ENV=1            keep an existing venv instead of rebuilding it
#
# Testing different package versions:
#   Add pins to .licenses/version-pins.txt (pip constraint syntax), e.g.
#       fairchem-core==2.5.0
#       mace-torch==0.3.14
#   They are applied as constraints to every install pass, then rerun this
#   script. Leave the file empty to use the versions pyproject.toml resolves.

set -euo pipefail

# repo root = parent of this script's dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-3.11}"
ENV_DIR="${ENV_DIR:-.venv-licenses}"
OUT_DIR=".licenses"
PINS_FILE="$OUT_DIR/version-pins.txt"
PY="$ENV_DIR/bin/python"

# conflict-free install passes; together they cover all extras + groups
EXTRA_PASSES=(
  ".[uma,aimnet,ase,pymatgen]"  # uma fork: fairchem-core, e3nn>=0.5, torch<2.9
  ".[mace]"                     # mace fork: e3nn==0.4.4
  ".[cu12]"                     # CUDA 12 stack (cuml-cu12, *-cu12, torch cu126)
  ".[cu13]"                     # CUDA 13 stack (cuml-cu13, *-cu13, torch cu130)
)
# NB: not "GROUPS" — that is a bash special variable (current user's GIDs).
DEP_GROUPS=(dev build docs distribution)

touch "$PINS_FILE"
CONSTRAINT_ARGS=(-c "$PINS_FILE")

if [[ "${REUSE_ENV:-0}" != "1" ]]; then
  echo ">>> creating fresh venv at $ENV_DIR (python $PYTHON)"
  rm -rf "$ENV_DIR"
  uv venv "$ENV_DIR" --python "$PYTHON"
fi

for spec in "${EXTRA_PASSES[@]}"; do
  echo ">>> installing $spec"
  uv pip install --python "$PY" "${CONSTRAINT_ARGS[@]}" "$spec"
done

echo ">>> installing dependency groups: ${DEP_GROUPS[*]}"
group_args=()
for g in "${DEP_GROUPS[@]}"; do group_args+=(--group "$g"); done
uv pip install --python "$PY" "${CONSTRAINT_ARGS[@]}" "${group_args[@]}" .

# pip-licenses runs from an isolated tool env and inspects $PY via --python,
# so pip-licenses' own deps never pollute the inventory.
PL=(uvx pip-licenses@4.1 --python "$PY")

# Collect the inventory once (with full license text), then render all three
# artifacts from it so manual license overrides apply consistently everywhere.
RAW_JSON="$(mktemp)"
trap 'rm -f "$RAW_JSON"' EXIT
echo ">>> collecting package inventory"
"${PL[@]}" --format=json --with-license-file --with-urls > "$RAW_JSON"

echo ">>> rendering summary.md / details.json / Third_party_attr.txt"
OVERRIDES="$OUT_DIR/license-overrides.json" \
"$PY" - "$RAW_JSON" "$OUT_DIR" <<'PYEOF'
import json, os, sys

data = json.load(open(sys.argv[1]))
out_dir = sys.argv[2]
norm = lambda n: (n or "").lower().replace("_", "-")

# load manual license determinations for packages with no usable SPDX metadata
overrides = {}
ov_path = os.environ.get("OVERRIDES")
if ov_path and os.path.exists(ov_path):
    for name, info in json.load(open(ov_path)).get("overrides", {}).items():
        overrides[norm(name)] = info

applied = []
for r in data:
    o = overrides.get(norm(r["Name"]))
    if not o:
        continue
    r["License"] = o["license"]  # short, scannable identifier
    if (r.get("LicenseText") or "UNKNOWN").strip() == "UNKNOWN":
        r["LicenseText"] = o.get("evidence", o["license"])  # paper trail, no file shipped
    applied.append(r["Name"])

data.sort(key=lambda r: (r["License"].lower(), r["Name"].lower()))  # mirror --order=license

# details.json — same schema/formatting pip-licenses emits
with open(os.path.join(out_dir, "details.json"), "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")

# summary.md — one scannable row per package; collapse any remaining full text
def short(v):
    v = (v or "").strip()
    if "\n" in v or len(v) > 60:
        return "<full text — see details.json>"
    return v or "UNKNOWN"
cols = ["Name", "Version", "License", "URL"]
rows = [[r["Name"], r["Version"], short(r["License"]), r.get("URL") or "UNKNOWN"] for r in data]
w = [max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(4)]
line = lambda v: "| " + " | ".join(v[i].ljust(w[i]) for i in range(4)) + " |"
with open(os.path.join(out_dir, "summary.md"), "w") as f:
    f.write(line(cols) + "\n")
    f.write("|" + "|".join("-" * (w[i] + 2) for i in range(4)) + "|\n")
    for r in rows:
        f.write(line(r) + "\n")

# Third_party_attr.txt — plain-vertical attribution dump (full license text)
field = lambda v: v if (v or "").strip() else "UNKNOWN"
blocks = ["\n".join([r["Name"], r["Version"], field(r["License"]),
                     field(r.get("LicenseFile")), field(r.get("LicenseText"))])
          for r in data]
with open(os.path.join(out_dir, "Third_party_attr.txt"), "w") as f:
    f.write("\n\n".join(blocks) + "\n")

# reviewer report — flag anything still unresolved so it can be added to overrides
print(f">>> done: {len(data)} packages inventoried; {len(applied)} override(s) applied")
unknown = [r for r in data if (r["License"] or "UNKNOWN").strip() == "UNKNOWN"]
fulltext = [r for r in data if "\n" in r["License"] or len(r["License"]) > 60]
if unknown:
    print(">>> still UNKNOWN (add to license-overrides.json):")
    for r in unknown: print(f"      {r['Name']} {r['Version']}")
if fulltext:
    print(">>> still full-text, not an SPDX id (add to license-overrides.json):")
    for r in fulltext: print(f"      {r['Name']} {r['Version']}")
PYEOF
