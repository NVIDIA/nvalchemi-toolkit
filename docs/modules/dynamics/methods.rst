.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _dynamics-methods:

=====================
Integrator Algorithms
=====================

This page documents exactly what each integrator and optimizer in
:ref:`dynamics-api` computes: the update scheme, the governing equations, and
the units of each control parameter. Every scheme is a thin ``nvalchemi``
wrapper (a :func:`torch.library.custom_op` for ``torch.compile`` / autograd
compatibility) around a GPU kernel in :mod:`nvalchemiops`; the kernels that
realise each method are collected in a collapsible panel at the end of every
section, so the implemented equations can be inspected directly.

.. note::

   **Units.** ``nvalchemi`` works in atomic-scale units throughout: length in
   angstrom (:math:`\mathrm{\AA}`), energy in electronvolt
   (:math:`\mathrm{eV}`), force in :math:`\mathrm{eV}/\mathrm{\AA}`, mass in
   daltons (amu), and time in femtoseconds (:math:`\mathrm{fs}`). Temperature is
   supplied in kelvin and enters the kernels as :math:`k_B T` in
   :math:`\mathrm{eV}`. Pressure and stress are in :math:`\mathrm{eV}/\mathrm{\AA}^3`
   (positive for compression; no bar/GPa conversion is applied — see
   :ref:`conventions`). Thermostat/barostat coupling times :math:`\tau` are in
   :math:`\mathrm{fs}` and the Langevin friction :math:`\gamma` in
   :math:`\mathrm{fs}^{-1}`.

Velocity Verlet (NVE)
=====================

The microcanonical integrator uses velocity Verlet, which is symplectic and
time-reversible and therefore conserves the total energy
:math:`H = \mathrm{KE} + \mathrm{PE}` to within integration error over long
trajectories. The step is split around the single force evaluation:

.. math::

   \mathbf{r}(t + \Delta t) &= \mathbf{r}(t) + \mathbf{v}(t)\,\Delta t
       + \tfrac{1}{2}\,\frac{\mathbf{F}(t)}{m}\,\Delta t^{2} \\
   \mathbf{v}(t + \tfrac{\Delta t}{2}) &= \mathbf{v}(t)
       + \tfrac{1}{2}\,\frac{\mathbf{F}(t)}{m}\,\Delta t \\
   \mathbf{v}(t + \Delta t) &= \mathbf{v}(t + \tfrac{\Delta t}{2})
       + \tfrac{1}{2}\,\frac{\mathbf{F}(t + \Delta t)}{m}\,\Delta t

The only control parameter is the timestep :math:`\Delta t` (fs); a typical
value is 0.5–1.0 fs. There is no thermostat, so any energy drift directly
measures model and integration error.

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.dynamics.integrators

   .. autofunction:: velocity_verlet_position_update
   .. autofunction:: velocity_verlet_velocity_finalize

Langevin dynamics (NVT, BAOAB)
==============================

The canonical-ensemble integrator uses the BAOAB splitting of Leimkuhler &
Matthews (2012), which gives high configurational-sampling accuracy. Each step
applies the sequence **B**-**A**-**O**-**A** before the force evaluation and a
final **B** after it, where **B** is a half force kick, **A** is a half drift,
and **O** is an Ornstein-Uhlenbeck velocity update that acts as an exact
thermostat:

.. math::

   \text{B:}\quad & \mathbf{v} \leftarrow \mathbf{v}
       + \tfrac{\Delta t}{2m}\mathbf{F} \\
   \text{A:}\quad & \mathbf{r} \leftarrow \mathbf{r}
       + \tfrac{\Delta t}{2}\mathbf{v} \\
   \text{O:}\quad & \mathbf{v} \leftarrow c_1\,\mathbf{v}
       + \sqrt{(1 - c_1^{2})\,\frac{k_B T}{m}}\;\boldsymbol{\xi},
       \qquad c_1 = e^{-\gamma\,\Delta t}

with :math:`\boldsymbol{\xi}` a standard normal draw. Control parameters are the
timestep :math:`\Delta t` (fs), the target temperature (K, entering as
:math:`k_B T`), and the friction :math:`\gamma` (:math:`\mathrm{fs}^{-1}`), which
sets the thermostat coupling strength. The stochastic **O** step is seeded by
``random_seed`` for reproducibility.

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.dynamics.integrators

   .. autofunction:: langevin_baoab_half_step
   .. autofunction:: langevin_baoab_finalize

Nose-Hoover chain thermostat (NVT)
==================================

The deterministic, time-reversible alternative for the canonical ensemble is a
Nosé-Hoover chain (NHC) using the Martyna-Tobias-Klein equations with
Yoshida-Suzuki factorisation. A chain of ``chain_length`` fictitious thermostat
variables (default 3) is coupled to the physical DOFs; the chain masses are set
from the coupling time :math:`\tau_T`:

.. math::

   Q_0 = N_\text{dof}\,k_B T\,\tau_T^{2},
   \qquad Q_k = k_B T\,\tau_T^{2}\ \ (k > 0)

Each step propagates the chain by a half step (rescaling the particle
velocities), performs the half velocity kick and full position drift, then
propagates the chain again after the force evaluation. Control parameters are
:math:`\Delta t` (fs), temperature (K), and :math:`\tau_T` (fs) — typically
10–100 :math:`\times \Delta t`. The scheme is ergodic for coupled systems but
can fail to thermalise stiff, near-harmonic modes, where Langevin is preferable.

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.dynamics.integrators

   .. autofunction:: nhc_compute_masses
   .. autofunction:: nhc_thermostat_chain_update
   .. autofunction:: nhc_velocity_half_step
   .. autofunction:: nhc_position_update
   .. autofunction:: nhc_compute_chain_energy
   .. autofunction:: nhc_compute_2ke

   A simpler stochastic-velocity-rescaling thermostat is also available:

   .. autofunction:: velocity_rescale

Isothermal-isobaric (NPT)
=========================

NPT couples the cell to a Martyna-Tobias-Klein barostat and the particles and
barostat to two independent Nosé-Hoover chains, sampling constant temperature
and pressure. The model must provide a ``stress`` output. The instantaneous
pressure tensor combines the kinetic and virial contributions,

.. math::

   \mathbf{P} = \frac{\mathbf{K} + \mathbf{W}}{V},

and the barostat inertia is fixed by the coupling time :math:`\tau_P`,

.. math::

   W = (N_f + d)\,k_B T\,\tau_P^{2}.

The cell strain rate :math:`\dot{\varepsilon}` is advanced by half steps from
the pressure imbalance,

.. math::

   \dot{\varepsilon}\ \mathrel{+}=\ \frac{\Delta t}{2}\,\frac{V}{W}\,
       (P_\text{inst} - P_\text{ext}),

interleaved with the two NHC updates, the particle velocity/position updates,
and the cell update. Control parameters: :math:`\Delta t` (fs), temperature (K),
target ``pressure`` (:math:`\mathrm{eV}/\mathrm{\AA}^3`), barostat time
:math:`\tau_P` (fs), and thermostat time :math:`\tau_T` (fs).
``pressure_coupling`` selects isotropic, anisotropic (orthorhombic), or
triclinic cell fluctuations.

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.dynamics.integrators

   .. autofunction:: compute_pressure_tensor
   .. autofunction:: compute_scalar_pressure
   .. autofunction:: compute_kinetic_tensor
   .. autofunction:: compute_cell_kinetic_energy
   .. autofunction:: compute_barostat_mass
   .. autofunction:: compute_barostat_potential_energy
   .. autofunction:: npt_thermostat_half_step
   .. autofunction:: npt_barostat_half_step
   .. autofunction:: npt_velocity_half_step
   .. autofunction:: npt_position_update
   .. autofunction:: npt_cell_update

Isenthalpic-isobaric (NPH)
==========================

NPH uses the same MTK barostat as NPT but omits the thermostat, so temperature
fluctuates and the enthalpy :math:`H = E + PV` is the conserved quantity. It is
useful for measuring the adiabatic response of a system to an applied pressure.
Control parameters are :math:`\Delta t` (fs), target ``pressure``
(:math:`\mathrm{eV}/\mathrm{\AA}^3`), and barostat time :math:`\tau_P` (fs).

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.dynamics.integrators

   .. autofunction:: nph_barostat_half_step
   .. autofunction:: nph_velocity_half_step

FIRE relaxation
===============

FIRE (Fast Inertial Relaxation Engine; Bitzek et al., 2006) is a geometry
optimiser, not a thermostatted integrator: it drives coordinates to a local
energy minimum using a damped-MD trajectory with an adaptive timestep. After
each force evaluation it computes the power :math:`P = \sum_i \mathbf{F}_i
\cdot \mathbf{v}_i` and mixes each velocity toward the force direction,

.. math::

   \mathbf{v} \leftarrow (1 - \alpha)\,\mathbf{v}
       + \alpha\,\sqrt{\frac{\mathbf{v}\cdot\mathbf{v}}
       {\mathbf{F}\cdot\mathbf{F}}}\;\mathbf{F}.

While :math:`P > 0` (moving downhill) the timestep grows and :math:`\alpha`
shrinks; when :math:`P \le 0` the velocity is zeroed and the timestep is cut.
Displacements are capped at ``maxstep`` (:math:`\mathrm{\AA}`) and the timestep
is clamped to ``[dt_min, dt_max]`` (fs). ``FIREVariableCell`` extends the same
mixing to the cell degrees of freedom using NPH-style cell propagation at zero
target pressure.

FIRE2 (Shuang et al., 2020) improves the restart conditions and the mixing
rule; it uses a distinct set of hyperparameters (``delaystep``, ``dtgrow``,
``dtshrink``, ``alpha0``, ...) and places the whole step before the force
evaluation.

.. dropdown:: Underlying ``nvalchemiops`` kernels

   .. currentmodule:: nvalchemiops.dynamics.optimizers.fire

   .. autofunction:: fire_step
   .. autofunction:: fire_update

   .. currentmodule:: nvalchemiops.torch.fire2

   .. autofunction:: fire2_step_coord
   .. autofunction:: fire2_step_coord_cell
