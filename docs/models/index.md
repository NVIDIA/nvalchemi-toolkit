<!-- markdownlint-disable MD013 MD014 -->

(models_section)=

# Supported Model Components

ALCHEMI Toolkit now exposes a composite calculator API through
{py:mod}`nvalchemi.models`. Instead of one wrapper interface, the package
provides explicit building blocks that can be combined into a pipeline:

- neighbor builders
- ML potentials
- direct physical terms
- derivative steps

For a step-by-step introduction, see the {ref}`models_guide`.

## Core Building Blocks

| Component | Purpose |
|---|---|
| {py:class}`~nvalchemi.models.CompositeCalculator` | Ordered pipeline runner that merges named outputs |
| {py:class}`~nvalchemi.models.EnergyDerivativesStep` | Derives forces and stresses from the current total energy |
| {py:class}`~nvalchemi.models.NeighborListBuilder` | Produces reusable external neighbor data |
| {py:class}`~nvalchemi.models.Potential` | Base class for energy and direct-output calculator steps |

## Machine-Learned Potentials

| Potential | Notes |
|---|---|
| {py:class}`~nvalchemi.models.MACEPotential` | External-neighbor MLIP. Advertises the minimum required cutoff and format. |
| {py:class}`~nvalchemi.models.AIMNet2Potential` | Internal-neighbor MLIP. Can expose `node_charges` for charge-coupled pipelines. |

## Physical / Classical Potentials

| Potential | Notes |
|---|---|
| {py:class}`~nvalchemi.models.DSFCoulombPotential` | Hybrid Coulomb term: direct positional derivatives plus charge-mediated autograd path. |
| {py:class}`~nvalchemi.models.DFTD3Potential` | Direct-output pairwise dispersion correction. |
| {py:class}`~nvalchemi.models.LennardJonesPotential` | Direct-output short-range repulsion/dispersion term. |
| {py:class}`~nvalchemi.models.EwaldCoulombPotential` | Periodic Coulomb term with `direct` or `autograd` derivative modes. |
| {py:class}`~nvalchemi.models.PMEPotential` | Periodic long-range Coulomb term with `direct` or `autograd` derivative modes. |

## Known Model Registry

Known artifact names can be resolved through the rewrite registry helpers:

- {py:func}`~nvalchemi.models.list_known_artifacts`
- {py:func}`~nvalchemi.models.resolve_known_artifact`

Typical usage:

```python
from nvalchemi.models import AIMNet2Potential, MACEPotential

aimnet2 = AIMNet2Potential(model="aimnet2")
mace = MACEPotential(model="mace-mp-0b3-medium")
```

The registry handles:

- name lookup
- download and verification
- optional post-processing
- cache reuse

## References

If you use any of the built-in model components provided by ALCHEMI Toolkit,
please cite the original publications for the underlying methods.

```{list-table}
:header-rows: 1
:widths: 20 80

* - Model
  - Citation
* - **MACE**
  - Batatia, I. *et al.* "MACE: Higher Order Equivariant Message Passing Neural
    Networks for Fast and Accurate Force Fields." *Advances in Neural Information
    Processing Systems (NeurIPS)*, 2022.
    [openreview.net/forum?id=YPpSngE-ZU](https://openreview.net/forum?id=YPpSngE-ZU)
* - **MACE-MP-0** (foundation)
  - Batatia, I. *et al.* "A foundation model for atomistic materials chemistry."
    *arXiv:2401.00096*, 2023.
    [doi:10.48550/arXiv.2401.00096](https://doi.org/10.48550/arXiv.2401.00096)
* - **AIMNet2**
  - Anstine, D. M., Zubatyuk, R. & Isayev, O. "AIMNet2: a neural network potential
    to meet your neutral, charged, organic, and elemental-organic needs."
    *Chem. Sci.* **16**, 10228--10244, 2025.
    [doi:10.1039/D4SC08572H](https://doi.org/10.1039/D4SC08572H)
* - **DFT-D3(BJ)**
  - Grimme, S. *et al.* "A consistent and accurate ab initio parametrization of
    density functional dispersion correction (DFT-D) for the 94 elements H-Pu."
    *J. Chem. Phys.* **132**, 154104, 2010.
    [doi:10.1063/1.3382344](https://doi.org/10.1063/1.3382344)
* -
  - Grimme, S., Ehrlich, S. & Goerigk, L. "Effect of the damping function in
    dispersion corrected density functional theory."
    *J. Comput. Chem.* **32**, 1456--1465, 2011.
    [doi:10.1002/jcc.21759](https://doi.org/10.1002/jcc.21759)
* - **Lennard-Jones**
  - Jones, J. E. "On the Determination of Molecular Fields."
    *Proc. R. Soc. Lond. A* **106** (738), 463--477, 1924.
    [doi:10.1098/rspa.1924.0082](https://doi.org/10.1098/rspa.1924.0082)
* - **Ewald Summation**
  - Ewald, P. P. "Die Berechnung optischer und elektrostatischer
    Gitterpotentiale." *Ann. Phys.* **369** (3), 253--287, 1921.
    [doi:10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)
* - **Particle Mesh Ewald**
  - Darden, T., York, D. & Pedersen, L. "Particle mesh Ewald: An
    N*log(N) method for Ewald sums in large systems."
    *J. Chem. Phys.* **98** (12), 10089--10092, 1993.
    [doi:10.1063/1.464397](https://doi.org/10.1063/1.464397)
```
