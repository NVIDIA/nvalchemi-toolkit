.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Models module (BaseModelMixin, ModelConfig, wrappers)
=====================================================

Every potential in ``nvalchemi`` ---machine-learned or classical---is exposed
through the same :class:`~nvalchemi.models.base.BaseModelMixin` interface and
described by a :class:`~nvalchemi.models.base.ModelConfig`, so it drops into any
:class:`~nvalchemi.dynamics.base.BaseDynamics` engine or training loop
unchanged. This page is a per-model reference: what each model is, how to
install it, and (for the classical models) the physics it implements. For the
wrapping API, composition patterns, and output conventions, see the
:doc:`models user guide </userguide/models>`.

.. currentmodule:: nvalchemi.models.base

Core classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ModelConfig
   NeighborConfig
   BaseModelMixin

Demo utilities
--------------

A minimal analytic model used throughout the tests and examples; useful as a
template when wrapping a new potential.

.. currentmodule:: nvalchemi.models.demo

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DemoModel
   DemoModelWrapper

Machine-learned potentials
--------------------------

Wrappers for third-party machine-learned interatomic potentials (MLIPs). Each
is an optional dependency installed through an extra; the wrapper adapts the
model's native inputs/outputs to the ``nvalchemi`` interface and exposes a
``from_checkpoint`` constructor that resolves named foundation-model checkpoints
or local files. Energies come from the underlying model and forces/stresses are
obtained by autograd on positions (conservative), unless the model provides them
directly.

MACE
~~~~

Equivariant message-passing network built on higher-order (E(3)-equivariant)
features. :class:`~nvalchemi.models.mace.MACEWrapper` accepts any MACE variant
(``MACE``, ``ScaleShiftMACE``, cuEquivariance-converted, or
``torch.compile``-d), builds a GPU one-hot ``node_attrs`` table to avoid
per-step CPU round-trips, and supports PBC via neighbor-list shifts.
:meth:`~nvalchemi.models.mace.MACEWrapper.from_checkpoint` loads local files or
named MACE-MP foundation models (e.g. ``"medium-0b2"``).

Install with the ``mace`` extra (the equivariance kernels live in the CUDA
group, so pair it with a ``cu`` extra)::

    uv sync --extra mace --extra cu13

.. currentmodule:: nvalchemi.models.mace

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MACEWrapper

AIMNet2
~~~~~~~

Message-passing network with explicit charge equilibration.
:class:`~nvalchemi.models.aimnet2.AIMNet2Wrapper` computes energy as the
primitive differentiable output (forces/stresses via autograd) and additionally
exposes partial charges and per-atom AIM feature embeddings. It declares an
external ``MATRIX``-format neighbor list at the model's AEV cutoff, satisfied by
a :class:`~nvalchemi.dynamics.hooks.NeighborListHook` (or the pipeline).
:meth:`~nvalchemi.models.aimnet2.AIMNet2Wrapper.from_checkpoint` resolves
checkpoints via ``AIMNet2Calculator``.

Install with the ``aimnet`` extra::

    uv sync --extra aimnet

.. currentmodule:: nvalchemi.models.aimnet2

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AIMNet2Wrapper

UMA
~~~

fairchem's Universal Models for Atoms — a single multi-task backbone with heads
for OMol25 (molecules), OMat24 (crystals), and related datasets.
:class:`~nvalchemi.models.uma.UMAWrapper` wraps a fairchem ``MLIPPredictUnit``
(the level at which energy/forces/stress are produced; the bare backbone yields
only embeddings). The task is fixed at construction and ``active_outputs``
reflects what it supports.
:meth:`~nvalchemi.models.uma.UMAWrapper.from_checkpoint` accepts registered
names (``"uma-s-1p1"``, ``"uma-s-1p2"``, ``"uma-m-1p1"``) or a local ``.pt``.

The ``uma`` (fairchem) stack conflicts with the CUDA/MACE dependencies, so
install it in a dedicated environment if you plan on using other MLIPs like MACE::

    UV_PROJECT_ENVIRONMENT=.venv-uma uv sync --extra uma --extra ase

.. currentmodule:: nvalchemi.models.uma

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   UMAWrapper

Physical / classical models
---------------------------

Analytic potentials backed by Warp GPU kernels in :mod:`nvalchemiops`. Like the
:ref:`integrator kernels <dynamics-methods>`, the equations each implements are
documented here so the physics is transparent. All work in the ``nvalchemi``
unit system (length :math:`\mathrm{\AA}`, energy :math:`\mathrm{eV}`, force
:math:`\mathrm{eV}/\mathrm{\AA}`, charge in units of :math:`e`) unless noted.

Lennard-Jones
~~~~~~~~~~~~~

A pairwise :math:`12`–:math:`6` potential summed over neighbor pairs within the
cutoff:

.. math::

   E = \sum_{i<j} 4\varepsilon\left[
       \left(\frac{\sigma}{r_{ij}}\right)^{12}
       - \left(\frac{\sigma}{r_{ij}}\right)^{6}\right].

An optional :math:`C^2`-continuous switching function tapers the energy and its
first two derivatives to zero over ``switch_width`` before the cutoff, avoiding
force discontinuities; ``switch_width=0`` gives a hard cutoff. Parameters:
:math:`\varepsilon` (well depth, eV), :math:`\sigma` (zero-crossing distance,
:math:`\mathrm{\AA}`), and ``cutoff`` (:math:`\mathrm{\AA}`).

.. currentmodule:: nvalchemi.models.lj

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   LennardJonesModelWrapper

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.interactions.lj

   .. autofunction:: lj_energy_forces
   .. autofunction:: lj_energy_forces_virial

DFT-D3 dispersion
~~~~~~~~~~~~~~~~~

Grimme's DFT-D3 dispersion correction with Becke-Johnson (BJ) damping — a
geometry-dependent :math:`C_6`/:math:`C_8` two-body dispersion term used to add
long-range van der Waals attraction to a DFT or MLIP base energy:

.. math::

   E_\text{disp} = -\sum_{i<j}\ \sum_{n=6,8}
       s_n\,\frac{C_n^{ij}}{r_{ij}^{n} + \left(a_1 R_0^{ij} + a_2\right)^{n}},
   \qquad R_0^{ij} = \sqrt{\frac{C_8^{ij}}{C_6^{ij}}} .

The coefficients :math:`C_6^{ij}` interpolate with the atomic coordination
number (controlled by ``k1``/``k3``), so the correction responds to the local
environment. Positions are supplied in :math:`\mathrm{\AA}` and converted to
Bohr internally; energies are returned in :math:`\mathrm{eV}`. Functional-specific
parameters ``a1`` (dimensionless), ``a2`` (Bohr), and ``s8`` (dimensionless) are
required; reference :math:`C_n`/:math:`R_0` tables load from a cached
``dftd3_parameters.pt``.

.. currentmodule:: nvalchemi.models.dftd3

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DFTD3ModelWrapper

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.torch.interactions.dispersion

   .. autofunction:: dftd3

Ewald summation
~~~~~~~~~~~~~~~

Exact long-range electrostatics for periodic systems, splitting the Coulomb sum
into a short-range real-space part and a smooth reciprocal-space part with a
Gaussian screening width :math:`\alpha`:

.. math::

   E = \underbrace{\frac{k_e}{2}\sum_{i \ne j}
         q_i q_j \frac{\operatorname{erfc}(\alpha r_{ij})}{r_{ij}}}_{\text{real space}}
     + \underbrace{\frac{k_e}{2V}\sum_{\mathbf{k}\ne 0}
         \frac{4\pi}{k^2}\,e^{-k^2/4\alpha^2}\,|S(\mathbf{k})|^2}_{\text{reciprocal space}}
     - \underbrace{\frac{k_e\,\alpha}{\sqrt{\pi}}\sum_i q_i^2}_{\text{self}},

with the structure factor :math:`S(\mathbf{k}) = \sum_i q_i e^{i\mathbf{k}\cdot\mathbf{r}_i}`
(a neutralising background term is added for charged cells). The real-space part
uses a neighbor matrix within ``cutoff``; :math:`\alpha` and the reciprocal
cutoff are chosen automatically from the requested ``accuracy``. The Coulomb
prefactor is :math:`k_e = 14.3996\ \mathrm{eV}\cdot\mathrm{\AA}/e^2`, and an
optional slab correction removes spurious periodic images along one axis for
2-D systems.

.. currentmodule:: nvalchemi.models.ewald

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   EwaldModelWrapper

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.torch.interactions.electrostatics.ewald

   .. autofunction:: ewald_real_space
   .. autofunction:: ewald_reciprocal_space

Particle-mesh Ewald (PME)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The same real/reciprocal Ewald decomposition, but the reciprocal sum is
evaluated by spreading charges onto a grid with B-spline interpolation and using
an FFT, giving :math:`O(N \log N)` scaling that is far cheaper than direct Ewald
for large systems. Parameters: ``cutoff`` (:math:`\mathrm{\AA}`), the mesh
resolution via ``mesh_spacing`` (:math:`\mathrm{\AA}`) or explicit
``mesh_dimensions``, and the B-spline ``spline_order`` (higher is smoother and
more accurate).

.. currentmodule:: nvalchemi.models.pme

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PMEModelWrapper

.. dropdown:: Underlying ``nvalchemiops`` kernels

   The real-space term reuses
   :func:`~nvalchemiops.torch.interactions.electrostatics.ewald.ewald_real_space`
   (documented above under Ewald); the reciprocal term is evaluated on the mesh
   via B-spline spread/gather:

   .. currentmodule:: nvalchemiops.torch.spline

   .. autofunction:: spline_spread
   .. autofunction:: spline_gather_with_force

Composition
-----------

Combine several models (e.g. an MLIP plus a long-range electrostatics term) into
a single potential; see the :doc:`models user guide </userguide/models>` for
wiring and neighbor-list sharing.

.. currentmodule:: nvalchemi.models.pipeline

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PipelineModelWrapper
   PipelineStep
   PipelineGroup
