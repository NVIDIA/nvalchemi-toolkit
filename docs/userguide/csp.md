(csp-guide)=

# Crystal structure prediction

Crystal structure prediction (CSP) explores possible arrangements of molecules
in a crystal. The `nvalchemi.csp` package provides four parts of this search:

- **Prepare the search:** assemble molecular conformers into a formula unit,
  estimate starting cell volumes, and choose compatible space groups. The
  molecule helpers use RDKit when available; raw tensor inputs are also supported.
- **Generate starting structures:** `CrystalPacker` chooses an input conformer
  for each independent molecule and places the molecules in a cell. It applies
  rigid-body translations and rotations to the molecules and can transform the
  cell to reduce intermolecular overlaps. The chosen conformers keep their
  internal geometries during packing.
- **Save and expand structures:** the Packer returns an asymmetric-unit (ASU)
  representation containing independent molecule placements, a cell, and
  symmetry information. Zarr storage retains these data without writing every
  atom in the cell. Selected structures can be expanded to full-cell Toolkit
  {py:class}`~nvalchemi.data.Batch` objects for physical optimization. Here
  “compact” describes the storage representation, regardless of cell volume.
- **Screen similar structures:** `RadialComparisonIndex` compares local atom
  environments to find possible duplicates after optimization. It also accepts
  fully periodic, partially periodic, and nonperiodic structures from other
  sources. See [comparison](#csp-comparison) for its limits and examples.

```{admonition} Packing does not rank crystals
:class: note

Packing produces starting geometries with intermolecular clashes resolved to
the specified overlap tolerance. Physical energies and polymorph ranking
require a suitable model and further optimization.
```

For API signatures and parameter details, see the {doc}`CSP API reference
<../modules/csp>`. See also the {ref}`data_guide` for Toolkit batches and the
{ref}`dynamics_guide` for optimization with a physical model.

(csp-water-example)=
## Generate starting structures

Prepare water with the optional RDKit helpers, request 200 structures on one
GPU, and expand two accepted structures. Install `nvalchemi-toolkit[rdkit]`
and use an environment with CUDA support.

```{note}
The 1000–1200 Å³ starting cells are deliberately oversized for one water
molecule so the example completes quickly. They are unsuitable starting
volumes for an ice search.
```

```python
import torch

from nvalchemi.csp.chem import (
    RDKitConformerConfig,
    build_molecular_packing_input,
    generate_conformers_from_smiles,
)
from nvalchemi.csp.packer import CrystalPacker, PackingConfig

molecule = generate_conformers_from_smiles(
    "O", config=RDKitConformerConfig(num_conformers=1, random_seed=7, num_threads=1)
)
packing_input = build_molecular_packing_input([molecule])
config = PackingConfig(
    z=1,
    z_prime=1,
    batch_size=64,
    max_candidates=1000,
    cell_volume_range=(1000.0, 1200.0),
    fixed_space_group=1,
)
packer = CrystalPacker(
    config=config,
    device="cuda:0",
)
result = packer(
    packing_input,
    num_samples=200,
    rng=torch.Generator(device="cuda:0").manual_seed(7),
    run_id=77,
)
print("accepted:", len(result), "trials:", result.generated_count)
print("stopped:", result.stop_reason)
if len(result) < 2:
    raise RuntimeError("Fewer than two candidates were accepted")

selected = result.structures.to_batch(indices=torch.tensor([0, 1]))
assert selected.num_graphs == 2
```

`num_samples` requests accepted structures. `batch_size` limits the trials
active at once, while `max_candidates` limits all initialized trials. Rejected
trials consume that budget, so a finite search may return fewer than 200
structures. Inspect the accepted count and stop reason before continuing.
`selected` contains full-cell atom coordinates for two candidates.

## Set up a molecular and crystallographic search

A **formula unit** is the ordered collection of molecules defining the
composition. A 1:1 hydrate, for example, has one solute and one water molecule
per formula unit. Pass those molecules in a fixed order, for example
`build_molecular_packing_input([solute, water])`.

Each molecule can have several input conformers. For a flexible molecule,
these represent alternative internal geometries that may pack differently.
The Packer chooses one conformer for each independently placed molecule, then
keeps its internal coordinates fixed while translating and rotating the
molecule and changing the cell. With `Z′ > 1`, it can choose different
conformers for the symmetry-independent copies of a molecule. The conformer
pool does not rank molecular energies. RDKit can generate the conformers, or
callers can supply coordinates prepared by another method.

### Choose Z, Z′, and a space group

`Z` counts formula units in the full conventional cell. `Z′` counts symmetry-independent
formula units in its asymmetric unit (ASU). For molecules in general positions,
the space group must have `Z / Z′` operations. For `Z=2` and `Z′=1`, familiar
choices include P-1 (group 2) and P2₁ (group 4), each with two operations.
For `Z=4` and `Z′=1`, four-operation choices include P2₁/c (group 14),
P2₁2₁2₁ (group 19), and Cc (group 9). With `Z′=2`, two formula units are
placed independently, so the Packer can choose different conformers for their
corresponding molecules. The bundled operations use the standard P2₁/c setting
for group 14; P2₁/n is an alternative cell setting of the same group.

Molecules on **special positions** are not supported. Some symmetry operations
map such a molecule onto itself, so expanding every independent molecule by
all operations would create duplicate copies. For example, a crystal with one
centrosymmetric molecule per P-1 cell has `Z=1` and `Z′=1/2`, which this
integer-`Z′` representation cannot express. To search for such an arrangement,
use a subgroup in which the molecule occupies a general position and choose
`Z` and `Z′` for that subgroup. P1 with `Z=Z′=1` represents this example; its
centrosymmetric conformer retains inversion under periodic repetition. In
general, a subgroup search does not enforce the omitted symmetry operations.

P-1 illustrates the two-operation case: identity leaves fractional coordinates
unchanged, and inversion maps `(x, y, z)` to `(-x, -y, -z)` periodically.

```python
from nvalchemi.csp import get_space_group_candidates, get_space_group_operations

assert 2 in get_space_group_candidates(2).tolist()
p_minus_1_operations = get_space_group_operations(2)
print(p_minus_1_operations)  # Identity and inversion.

# Reuse the water input with P-1 for a small second run.
p_minus_1_result = packer(
    packing_input,
    num_samples=2,
    rng=torch.Generator(device="cuda:0").manual_seed(8),
    z=2,
    z_prime=1,
    fixed_space_group=2,
)
```

### Choose starting volume and sampling weights

Choose starting volumes from molecular size and an expected crystal density.
{py:func}`~nvalchemi.csp.estimate_formula_unit_volume` returns an estimate
`V_fu` for one formula unit in Å³. `cell_volume_range=(low, high)` specifies
absolute **cell** volumes in Å³. Alternatively,
`cell_volume_scale_range=(low, high)` uses `(low * Z * V_fu, high * Z * V_fu)`.
Supply exactly one range. Both control initial cells. Packing can shrink a
sampled cell but does not expand its volume; later physical optimization can
change the volume in either direction.

A fixed group gives one symmetry per run. Without `fixed_space_group`, the
Packer samples compatible groups using weights derived from the Cambridge
Structural Database (CSD). A caller-supplied `space_group_probabilities`
mapping **replaces** those weights; omitted groups have zero weight, and
remaining compatible positive weights are normalized. The bundled weights
come from the [CCDC space-group statistics report dated 1 January 2026](https://www.ccdc.cam.ac.uk/media/CSD-Space-Group-Statistics-Space-Group-Number-Ordering-2026.pdf).

```python
from nvalchemi.csp import sample_space_groups

# Draw groups with four operations using the bundled CSD weights.
csd_draw = sample_space_groups(5, num_operations=4, seed=7)
# Restrict custom draws to group 14 and P2₁2₁2₁ (group 19).
custom_draw = sample_space_groups(
    5, num_operations=4, probabilities={14: 1.0, 19: 1.0}, seed=7
)
print(custom_draw.tolist())
```

Use `get_space_group_candidates()` to inspect compatible groups before a run.
The helpers also filter by crystal system or by Sohncke groups, whose
operations preserve molecular handedness. `sohncke_only=True` is appropriate
when symmetry copies of a chiral molecule must keep their handedness. Call-time
Packer overrides, as in the P-1 example, revalidate a complete setting without
changing the base config. Fixed-group settings cannot be combined with sampled
weights. Use separate calls to request an accepted-structure target for each
Z/Z′ or space-group choice; weighted sampling does not guarantee coverage.
When changing a fixed-group Packer to weighted sampling, clear the inherited
setting with `fixed_space_group=None`. Likewise, clear `cell_volume_range`
before supplying `cell_volume_scale_range`.

### Understand acceptance and shortfall

The Packer checks contacts between different molecular copies, including
periodic copies. An **overlap** is the positive difference between a contact
cutoff and an atom-pair distance, in Å. Small residual overlap can remain: a
trial is accepted when its largest overlap is at most `overlap_tolerance`.
The accepted structures carry `steps`, `total_overlap`, and `max_overlap` in
`result.structures.properties`. These are clash diagnostics, not physical
energies.

Contacts are checked at iteration zero, at each
`convergence_check_interval`, and when a trial expires after
`max_steps_per_candidate` updates. An optional progress callback reports
active-trial diagnostics. If the finite candidate budget ends first, the
result can contain fewer accepted structures than requested. Restrictive
cell-shape or volume limits can instead prevent initialization and raise
`RuntimeError`. See {py:class}`~nvalchemi.csp.packer.PackingConfig` for the
movement, cell-shape, and progress controls.

## Distribute candidate generation

Independent packing trials can run across several GPUs, usually with one
process per GPU. Every process needs equivalent formula-unit input and an
agreed global target and trial budget. This distributes packing work;
distributing later optimization stages is a separate Toolkit workflow (see
{ref}`dynamics_guide`).

The application creates the process group and assigns a local CUDA device to
each process. This fragment assumes a `torchrun` launch and an initialized
NCCL group. Prepare the [input and config](#csp-water-example) on each
process; run the packing call below in place of the single-GPU call:

```python
import os
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
device = f"cuda:{local_rank}"
torch.cuda.set_device(local_rank)
rank_packer = CrystalPacker(config=config, device=device)
rank_result = rank_packer(
    packing_input,
    num_samples=200,
    rng=torch.Generator(device=device).manual_seed(7),
    process_group=dist.group.WORLD,
    gather_to_rank=0,
    run_id=78,
)
if dist.get_rank() == 0:
    assert rank_result is not None
else:
    assert rank_result is None
```

All ranks provide equivalent formula-unit input and agree on the global target
and candidate budget. The default quotas divide them as evenly as possible;
custom `rank_targets` must sum to the global target, and with a finite budget
require `rank_candidate_budgets` summing to that budget. With
`gather_to_rank=0`, rank 0 receives the combined ASU representations and the
others receive `None`; with `None`, each rank retains its local result. The
caller owns process launch, group lifetime, device assignment, and output
storage. Gloo with CPU Packers is also supported.

## Store and relax selected candidates

The examples below continue from the [single-GPU result](#csp-water-example).
For a distributed run, save on the gather destination using its non-`None`
`rank_result`.

{py:class}`~nvalchemi.csp.RigidMoleculeASUBatch` stores independent molecule
placements, shared conformers, cell, and space-group information instead of
all full-cell atom coordinates. Save accepted ASU representations first, then
expand only the candidates chosen for expensive physical optimization.

```python
from nvalchemi.csp import CSPZarrReader, CSPZarrWriter

with CSPZarrWriter("water-candidates.zarr") as writer:
    writer.write(result.structures)

with CSPZarrReader("water-candidates.zarr") as reader:
    asu_rows = reader.read(indices=torch.tensor([0, 1]))
    batch = reader.read_batch(indices=torch.tensor([0, 1]), device="cuda:0")
assert asu_rows.num_structures == batch.num_graphs == 2
```

`write()` creates a new store and refuses to replace an existing path. Use a
fresh path when rerunning the example. `append()` adds compatible rows; an
identical active structure ID is skipped, while conflicting data for that ID
raises an error. `read_batch()` requires explicit row indices so a large store
is not expanded accidentally. An empty Packer result is a valid ASU batch,
but handle it before passing data to an optimizer.

The expanded `batch` can enter Toolkit dynamics. This one-step demonstration
shows the handoff; `DemoModel` is not a physical crystal potential and its
output has no CSP ranking value. For a search, use a model suitable for the
composition and intermolecular interactions, configure its required neighbor
data, and relax to an appropriate convergence criterion (see
{ref}`dynamics_guide`).

```python
from nvalchemi.dynamics import FIRE2
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

demo_model = DemoModelWrapper(DemoModel()).to("cuda:0")
with FIRE2(model=demo_model, dt=0.05, n_steps=1) as optimizer:
    demo_relaxed = optimizer.run(batch)
```

```{admonition} Space-group symmetry during optimization
:class: warning

Atomistic optimization does not enforce the generated space group. The stored
space-group field records the source packing; it does not describe the symmetry
of the optimized coordinates. See {ref}`source metadata <csp-source-metadata>`
for the retained provenance fields.
```

In a single-process run, each accepted structure has a stable
`[run_id, accepted_structure_number]` ID, with the second number starting at
zero. Distributed runs use rank-strided numbers to keep IDs unique. Expansion
carries these IDs and the source space group into the atomistic Batch.

### Run packing in a generation pipeline

Use `AtomisticGenerator` when packing is the first stage of a workflow that
continues with Toolkit dynamics. The Packer returns accepted structures in an
ASU representation; FIRE2 needs a full-cell {py:class}`~nvalchemi.data.Batch`.
The generating function expands the accepted structures before returning
them. This example reuses `packer`, `packing_input`, and `demo_model` from
above:

```python
from nvalchemi.gen import AtomisticGenerator, GenerationPipeline


def generate_for_relaxation(formula_unit, *, num_samples, rng):
    """Pack a formula unit and expand accepted structures for dynamics."""
    packed = packer(formula_unit, num_samples=num_samples, rng=rng)
    if len(packed) == 0:
        raise RuntimeError("No packing candidate was accepted")
    return packed.structures.to_batch()


generator = AtomisticGenerator(
    generator_func=generate_for_relaxation,
    num_samples=4,
    required_inputs=frozenset(),
    outputs=frozenset({"positions", "atomic_numbers", "cell", "pbc"}),
    device="cuda:0",
    seed=79,
)
pipeline = GenerationPipeline(
    stages=[generator, FIRE2(model=demo_model, dt=0.05, n_steps=1)]
)
with pipeline:
    relaxed = pipeline(packing_input)
```

The generator requests four accepted structures and relaxes all those returned
by the Packer. A finite trial budget may yield fewer, so the function checks
for an empty result. The pipeline passes `packing_input` to the generating
function. The empty `required_inputs` declaration means no Batch fields are
required from that input, while `outputs` names fields in the returned Batch.
`AFTER_GENERATE` hooks see the expanded Batch, and `GenerationPipeline` passes
it to FIRE2. Returning the raw
`PackingResult` instead would bypass Batch hooks and could not feed a dynamics
stage.

(csp-comparison)=
## Screen similar and duplicate structures

A CSP campaign can generate and optimize $10^4$ to $10^6$ trial structures,
many of which may converge to similar geometries. A pool of $N$ structures
has $N(N-1)/2$ possible unordered pairs: about 50 million for $10^4$
structures and 500 billion for $10^6$. The comparison API makes each
first-pass comparison fast; exhaustive pair enumeration still grows
quadratically. For very large pools, select plausible pairs or work in chunks.

### What the radial mismatch score means

For each atom, the index sorts distances to neighbors within `cutoff`, in Å;
shorter lists are padded with the cutoff distance. It finds the most similar
compatible atom environment in the other structure by comparing those
distance patterns, regardless of the atoms' coordinates in a common frame.
Every atom environment must find a match in the other structure, and the
comparison runs in both directions. Several atoms may match the same
environment: this is not a one-to-one atom assignment.

The **radial mismatch score** is the largest relative mismatch remaining
after these matches.
For two distances, the mismatch is `max(d1, d2) / min(d1, d2) - 1`: 2.0 Å and
2.1 Å differ by 5%. Thus `threshold=0.05` allows about 5% mismatch. It is a
dimensionless relative threshold, not a 0.05 Å tolerance. Lower scores mean
closer local distance patterns.

The score ignores angles, chirality, energy, and distances beyond `cutoff`.
Distinct crystals can therefore have low mismatch scores: **matches can be false
positives**. For molecular crystals, [OXtalign](https://github.com/OXtal/oxtalign)
compares molecular packing; [pymatgen's StructureMatcher](https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure_matcher.StructureMatcher)
offers general crystal-structure matching. These are external options for
confirming important matches. With `deduplicate(confirm=...)`, the caller
supplies a more detailed comparison function. CSP passes it the pairs that
survive the radial screen and uses the confirmed subset to choose retained
representatives. No final crystallographic comparator is bundled here.
The index honors fully periodic, partially periodic, or nonperiodic coordinates
according to the Batch cell and PBC flags. Periodic directions require both a
cell and PBC flags; a cell alone does not enable periodic comparison.

### Choose how atom types restrict matches

Each distance list belongs to a **central atom** and describes its neighbors.
An untyped first screen lets any central atoms match based on distances alone;
it avoids type preparation and uses the simplest descriptor, but can leave
more false positives. Supplying `atom_types` restricts which central atoms may
match. Setting `typed_neighbors=True` also separates their neighbor distances
by atom type. Callers can choose the level of chemical selectivity:

| Settings | Which atoms may match? |
| --- | --- |
| `atom_types=None` | Any central atoms; elements are not inferred. |
| `atom_types=batch.atomic_numbers`, `typed_neighbors=False` | Central atoms of the same element; all neighbor distances remain in one list. |
| Topological `atom_types`, `typed_neighbors=False` | Central atoms with the same molecular connectivity role; neighbor distances remain in one list. |
| Types with `typed_neighbors=True` | Central atoms of the same type, with neighbor distances compared separately by neighbor type. |

Atomic numbers alone do not identify an atom's role within a molecule. In
ethanol, both the methyl and methylene carbons have atomic number 6 but
different bond connections. **Topological atom types** distinguish those
roles. With explicit hydrogens, the three methyl hydrogens share one type,
the two methylene hydrogens another, and the hydroxyl hydrogen has its own.
The helpers group atoms that can be exchanged by relabeling a molecule while
preserving elements and bond connections.

Topological type numbers are local labels. Build one
{py:class}`~nvalchemi.csp.chem.TopologicalAtomTypeMap` from the formula unit's
ordered atomic numbers and bonds, then use it to label both relaxed candidates
and any experimental structures compared with them. The labels ignore bond
order, charge, isotope labels, stereochemistry, geometry, and implicit
hydrogens. For a single graph, use
{py:func}`~nvalchemi.csp.chem.topological_atom_types_from_connectivity` with
tensors or {py:func}`~nvalchemi.csp.chem.topological_atom_types_from_mol` with
RDKit. The default `typed_neighbors=True` separates neighbor lists when types
are supplied. Callers can run successive typing screens on surviving pairs;
the Toolkit does not perform that sequence automatically.

### Deduplicate a relaxed structure pool

After the generate → optimize workflow above, collect relaxed structures
generated from the same molecular input into a standard Toolkit `Batch`,
called `batch` below.
Assume optimization has preserved molecular connectivity: the structures may
have different geometries, but they share the same molecular topology. Retain
the molecular-bond matrix as `template_adjacency`, using the **same atom order**
as `packing_input.atomic_numbers`. Packer expansion records each atom's source
index, which the type map uses to label relaxed structures even when they have
different `Z` or `Z′`. Keep that index aligned with its atom during optimization
and storage. The map checks indices and elements, but cannot establish that a
loaded Batch came from this particular formula unit. See
{ref}`csp-source-metadata` for the provenance field.

Topological types are optional for approximate matching. An index without
types compares local distances regardless of atom identity; using atomic
numbers as types restricts central atoms to the same element. Either may leave
more false-positive matches than topology-based screening, because atoms of
the same element can play different roles in a molecule.

For a pool with a manageable number of possible pairs, refine matches in three
steps: distances alone, then the molecular role of the central atom, then the
roles of both the central atom and its neighbors. Each step keeps only pairs
that passed the preceding one. Release each index before building the next.
Replace the `...` placeholders with your formula-unit input, its bond matrix,
and a relaxed Toolkit `Batch` from the same search:

```python
from nvalchemi.csp.chem import TopologicalAtomTypeMap
from nvalchemi.csp.comparison import RadialComparisonIndex

packing_input = ...  # Formula-unit input used to generate this search pool.
template_adjacency = ...  # Boolean bonds in packing_input.atomic_numbers order.
batch = ...  # Toolkit Batch of optimized trial crystals with Packer source fields.

cutoff = 15.0  # Å
threshold = 0.05  # maximum relative distance mismatch

# 1. Compare local distances without atom types.
untyped_index = RadialComparisonIndex.build(batch, cutoff=cutoff)
pairs = untyped_index.find_matches(threshold=threshold)
del untyped_index

# 2. For pairs marked as possible duplicates by step 1, require the
#    central atoms to have the same molecular role.
type_map = TopologicalAtomTypeMap(
    packing_input.atomic_numbers, template_adjacency
)
topology_types = type_map.for_batch(batch)
center_index = RadialComparisonIndex.build(
    batch, cutoff=cutoff, atom_types=topology_types, typed_neighbors=False
)
pairs = center_index.find_matches(threshold=threshold, pair_indices=pairs)
del center_index

# 3. For pairs marked as possible duplicates by step 2, also compare
#    neighbors by their molecular roles.
typed_index = RadialComparisonIndex.build(
    batch, cutoff=cutoff, atom_types=topology_types, typed_neighbors=True
)
approx_duplicates = typed_index.find_matches(
    threshold=threshold, pair_indices=pairs
)
approx_scores = typed_index.score_pairs(approx_duplicates)
```

Each row of `approx_duplicates` identifies two structures with similar
local distance patterns; `approx_scores` is aligned with those rows. These
are approximate matches, not confirmed duplicate crystals. This example
collects the pairs from each stage, so the pair tensor itself can grow large.
If you only need representatives, build just the `typed_index` from step 3
and skip the first two indices and pair lists:

```python
result = typed_index.deduplicate(threshold=threshold)
unique_indices = result.retained_indices
deduplicated_batch = batch[unique_indices]
```

`deduplicated_batch` contains the provisional representatives; the other
structures are assigned to them in `result.representative_indices`. If you
need confirmed structural equivalence before discarding a candidate, pass a
more specific comparison as `deduplicate(confirm=...)`. The callback receives
proposed pairs and returns only confirmed ones in their original pair order.
Confirming after deduplication cannot recover a structure already skipped by
its greedy pass. Within each typed screen, type IDs must have the same meaning
for every structure.

`deduplicate()` is greedy: it visits structures in input order and assigns each
candidate to the first earlier **retained** representative that matches it. A
match at the chosen threshold is not necessarily transitive: A may match B and
B may match C even though A does not match C. In the order A, B, C, the
algorithm retains A, assigns B to A, then retains C because it compares C with
A, not with discarded B. In the order B, A, C, it retains B and assigns both A
and C to B. The chosen representatives therefore depend on input order.
Structures assigned to the same representative need not match one another. To
favor low-energy representatives, order the input Batch by energy before
building an index; the returned indices then refer to that order. For a larger
pool, process chunks in input order: combine previously retained
representatives with the next chunk, deduplicate that combined Batch, and carry
its representatives forward. Preserve representative order and carry original
pool indices alongside each combined Batch; each result's indices refer to that
combined Batch. Use the same threshold and confirmation rule at every step.
This limits the resident index to retained structures plus one chunk, at the
cost of rebuilding it each time. If nearly every structure is unique, the
retained set can still grow large. Deduplicating chunks independently and
merging their survivors can change the result because similarity need not be
transitive. Build new indices after geometry or atom labels change. See the
{py:class}`~nvalchemi.csp.comparison.RadialComparisonIndex` reference for score
dtype and CUDA memory settings.

### Compare with an experimental structure

Here, `batch` contains relaxed Packer structures from `packing_input`, and
`template_adjacency` contains formula-unit bonds in
`packing_input.atomic_numbers` order.

An experimental CIF supplies cell and coordinates, but molecular bonds still
need to be established. Resolve disorder or partial occupancy, use the same
hydrogen choices as the candidate pool, and check that ASE reads a complete
unit cell. Also check that the proportions of molecular components match the
formula unit: a 1:1 two-component formula may appear as 2:2 in a cell, but
not 2:1. Install `nvalchemi-toolkit[ase]` for CIF loading.
Toolkit does not provide `connectivity_from_reference`; implement it for your
CIF preparation workflow. It must return two Torch tensors in ASE atom order:
integral atomic numbers `[N]` and a symmetric boolean molecular-bond matrix
`[N, N]` with a zero diagonal. Bonds across cell boundaries must be included;
nearby atoms are not necessarily bonded.

Build the type map from the **same ordered formula unit** used for the search.
It matches complete molecular bond graphs to the formula unit, assigning the
same numeric type labels even when the CIF lists atoms in a different order.
Unknown molecular components are rejected. Typing does not check component
proportions or infer missing bonds. If you ran the preceding deduplication
example, reuse its `type_map` and `typed_index` instead of rebuilding them.
For this standalone snippet, replace the `...` placeholders with the same
formula-unit input, bond matrix, and relaxed Batch.

```python
from ase.io import read

from nvalchemi.csp.chem import TopologicalAtomTypeMap
from nvalchemi.csp.comparison import RadialComparisonIndex
from nvalchemi.data import AtomicData, Batch

packing_input = ...  # The same formula-unit input used for the candidate pool.
template_adjacency = ...  # Its bonds in packing_input.atomic_numbers order.
batch = ...  # Toolkit Batch of optimized trial crystals with Packer source fields.

cutoff = 15.0  # Å
threshold = 0.05
type_map = TopologicalAtomTypeMap(
    packing_input.atomic_numbers, template_adjacency
)
typed_index = RadialComparisonIndex.build(
    batch, cutoff=cutoff, atom_types=type_map.for_batch(batch), typed_neighbors=True
)

reference_atoms = read("experimental.cif")
reference_numbers, reference_adjacency = connectivity_from_reference(
    reference_atoms
)
reference_types = type_map.for_connectivity(
    reference_numbers, reference_adjacency
)
reference_batch = Batch.from_data_list(
    [AtomicData.from_atoms(reference_atoms, device=typed_index.device)]
)
reference_index = RadialComparisonIndex.build(
    reference_batch,
    cutoff=cutoff,
    atom_types=reference_types,
    typed_neighbors=True,
)
experimental_matches = typed_index.find_matches(
    other=reference_index, threshold=threshold
)
experimental_scores = typed_index.score_pairs(
    experimental_matches, other=reference_index
)
```

Each row of `experimental_matches` is `[candidate_index, 0]`: column zero
identifies a structure in the original search pool, and zero identifies the
single experimental reference. `experimental_scores` has one mismatch score
per returned pair. A low score proposes a match for more detailed structural
comparison; it does not establish that the structures are crystallographically
equivalent. For additional CIFs, repeat the reference preparation and matching
block with the same `type_map` and `typed_index`.

## Advanced data and storage details

The formula-unit input owns its conformer pool, contact distances in Å,
component IDs, and volume estimate in Å³. An ASU batch shares that input and
stores each candidate's cell, space group, and independent placements. Cell
vectors are rows of a 3 × 3 matrix in Å. Fractional molecular centers are
dimensionless and become Cartesian through `fractional @ cell`; rigid
rotations act on Cartesian molecular displacements. Expansion emits
`A_fu * Z` atoms per cell, where `A_fu` is the atom count of one formula unit.
Molecular centers are wrapped; atom coordinates are not wrapped separately.

Tensor-only callers can construct {py:class}`~nvalchemi.csp.MolecularPackingInput`
and {py:class}`~nvalchemi.csp.RigidMoleculeASUBatch` directly. Pointer arrays
mark spans in the next array; `[0, 2]` means the first item occupies indices
0 and 1. See their API pages for shapes and validation rules. Direct callers
must supply valid rotation matrices and cells compatible with the selected
space groups.

(csp-source-metadata)=
### Source metadata after geometry changes

Full-cell expansion records where every atom and structure came from. These
fields are provenance, even if subsequent optimization breaks source
symmetry:

| Field | Alignment | Meaning |
| --- | --- | --- |
| `csp_source_asu_atom_index` | Atom | Atom in the repeated ASU order |
| `csp_source_molecule_index` | Atom | Molecule in the repeated ASU order |
| `csp_source_component_index` | Atom | Component from the formula unit |
| `csp_source_conformer_index` | Atom | Conformer in the shared input |
| `csp_source_symmetry_operation_index` | Atom | Operation that generated the copy |
| `csp_source_space_group` | System | Source International group number |
| `csp_source_z` | System | Source formula units in the cell |
| `csp_source_z_prime` | System | Source formula units in the ASU |
| `csp_source_structure_id` | System | int64 `[run_id, accepted_structure_number]` |

For `Z′ > 1`, ASU atom and molecule indices include every repeated independent
formula unit. Toolkit Batch selection and cloning preserve aligned fields.
To type a saved relaxed Batch later, retain the ordered formula-unit atomic
numbers and bond matrix once alongside the search results and rebuild its
`TopologicalAtomTypeMap`. Source indices alone do not identify the molecular
graph. Keep pools from different formula-unit inputs separate when assigning
or comparing type IDs.

### Store maintenance and reproducibility

`read()` and `delete()` address the current undeleted row order; deleting row
0 makes old row 1 the new row 0 without changing IDs. Existing readers must
`refresh()` after writes. Use one active writer per store and reopen it if
another writer changes the store. `defragment()` reclaims deleted-row space
and needs exclusive access. If local-directory replacement is interrupted
after moving the original, restore the sibling `.NAME.csp-backup-*` directory
before reopening. Custom property names cannot contain `/` or be `.`, `..`,
or `zarr.json`.

Without an explicit run ID, the Packer generates one independently of the
Torch RNG. Distributed IDs are `[run_id, group_rank + group_size *
local_accepted_number]`; gathered rows follow rank order. For replay, retain
the run ID, each rank's seed and settings, formula input, group size, and rank
mapping. Changing group membership can change candidates. Catchable failures
on live ranks are coordinated at communication points; process termination
cannot be coordinated by the Packer.
