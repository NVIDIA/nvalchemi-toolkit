# Third-party license inventory

These artifacts are generated for open-source legal review. They cover **every**
optional dependency and dependency group, including the mutually-exclusive
extras (`cu12`, `cu13`, `mace`, `uma`).

## Files

| File | Purpose |
|------|---------|
| `summary.md` | Human-readable table: name, version, license, URL (one row per package) |
| `details.json` | Machine-readable; `License` plus the full license text in `LicenseText` |
| `Third_party_attr.txt` | Attribution dump: name / version / license / file / full text |

`pip-licenses` collects the raw inventory once (`--format=json --with-license-file
--with-urls`); the script then renders all three files from that single dataset so
manual license overrides (below) apply consistently everywhere.

## Regenerating

```bash
.licenses/generate_licenses.sh
```

This builds a throwaway venv (`.venv-licenses/`), force-installs all extras and
groups into it, and rewrites the three files above. It prints the total package
count and flags any package whose license is `UNKNOWN`.

### Why a throwaway, non-functional env?

The extras conflict by design and cannot coexist in a working install:

- `cu12` vs `cu13` — different CUDA toolchains
- `mace` (pins `e3nn==0.4.4`) vs `uma`/fairchem-core (needs `e3nn>=0.5`)
- `uma` (caps `torch<2.9`) vs the `cuXX` stacks (floor `torch>=2.11`)

For a license inventory we only need each package's *name* and *license*, not a
runnable environment. `uv pip install` does not prune, so installing each
conflict-free group in its own pass accumulates the union of all packages.
Same-named conflicting packages settle on whichever version installed last,
which does not affect the license.

## Fixing missing / unhelpful licenses

Some packages expose no usable SPDX identifier — their `License` metadata is
empty, `UNKNOWN`, or the entire license text (e.g. NVIDIA SDK EULAs). The script
flags these in its output under "still UNKNOWN" / "still full-text".

Record the correct license in `license-overrides.json` and rerun; the override
is applied to all three files. The full license text (when shipped) is preserved
in `details.json`/`Third_party_attr.txt` regardless. Each entry carries an
`evidence` field documenting how the license was determined (bundled LICENSE
file, EULA URL, etc.) — keep this current for the legal audit trail.

```json
{
  "overrides": {
    "torch-ema": { "license": "MIT", "evidence": "bundled LICENSE: MIT text" }
  }
}
```

## Testing specific versions

Add [pip constraint](https://pip.pypa.io/en/stable/user_guide/#constraints-files)
lines to `version-pins.txt`, then rerun the script:

```
fairchem-core==2.5.0
mace-torch==0.3.14
```

The constraints are applied to every install pass.

## Overrides

```bash
PYTHON=3.12 .licenses/generate_licenses.sh   # change interpreter version
REUSE_ENV=1 .licenses/generate_licenses.sh   # skip the venv rebuild
```
