Advanced Examples
=================

These examples are for users who want to extend the nvalchemi-toolkit
framework.  They require understanding of the intermediate tier.

**01 — Biased Potential**: BiasedPotentialHook for harmonic COM restraints
and umbrella sampling patterns.

**02 — Custom Hook**: Implementing the Hook protocol with a full
radial distribution function accumulator.

**03 — Custom Convergence**: ConvergenceHook with multiple criteria and
custom_op for arbitrary convergence logic.

**04 — MACE NVT**: Using a real MACE MLIP for NVT dynamics; automatic
neighbor list wiring via ModelCard; LJ fallback for CI.

**05 — Custom Integrator**: Subclassing BaseDynamics to implement a
velocity-rescaling thermostat; the pre_update/post_update contract;
_init_state for stateful integrators.
