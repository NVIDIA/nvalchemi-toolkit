<!-- markdownlint-disable MD014 -->

(models_section)=

# Supported Models

ALCHEMI Toolkit ships wrappers for several machine-learning interatomic
potentials (MLIPs) and classical force fields.  Every wrapper implements the
{py:class}`~nvalchemi.models.base.BaseModelMixin` interface and exposes a
{py:class}`~nvalchemi.models.base.ModelCard` that declares its capabilities
and input requirements.

For a step-by-step guide on wrapping your own model, see the
{ref}`models_guide`.

## Machine-Learned Potentials

Neural-network potentials that learn interatomic interactions from quantum
mechanical reference data.

```{eval-rst}
.. model-capability-table::
   :category: ml
```

## Physical / Classical Models

Analytical force fields and correction terms based on known physical
functional forms.

```{eval-rst}
.. model-capability-table::
   :category: physical
```

```{note}
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` is excluded from
the tables above because its capabilities are **synthesized** at runtime from
the sub-models it composes (see
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper`).
All tables are **auto-generated** from each wrapper's
{py:class}`~nvalchemi.models.base.ModelCard` at documentation build time.
```

## Model Composition

Models can be combined using the ``+`` operator for simple additive
composition, or the explicit
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` API for
dependent pipelines with shared autograd groups and inter-model wiring.
See {ref}`models_guide` for full details.

```python
# Simple: sum energies, forces, stresses
combined = mace_model + dftd3_model

# Advanced: dependent pipeline with shared autograd
from nvalchemi.models.pipeline import PipelineModelWrapper, PipelineGroup, PipelineStep

pipe = PipelineModelWrapper(groups=[
    PipelineGroup(
        steps=[PipelineStep(aimnet2, wire={"charges": "node_charges"}), ewald],
        use_autograd=True,
    ),
    PipelineGroup(steps=[dftd3]),
])
```

## Foundation Models

Pre-trained checkpoints that can be loaded directly via
{py:func}`~nvalchemi.models.registry.list_foundation_models` and
``MACEWrapper.from_checkpoint(name)``.

```{eval-rst}
.. foundation-model-table::
```

## References

If you use any of the model wrappers provided by ALCHEMI Toolkit, please cite
the original publications for the underlying methods.

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
