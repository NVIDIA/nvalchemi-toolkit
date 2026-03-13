Basic Examples
==============

These examples introduce the core nvalchemi-toolkit workflow to users
familiar with tools like ASE.  Each script focuses on one feature set
and runs end-to-end on a single GPU in under 60 seconds.

**01 — Data Structures**: AtomicData and Batch API.
**02 — Geometry Optimization**: FIRE optimizer with NeighborListHook and ConvergenceHook.
**03 — ASE Integration**: Loading ASE structures, FreezeAtomsHook on a surface system.
**04 — NVE MD**: Microcanonical dynamics, WrapPeriodicHook, EnergyDriftMonitorHook.
**05 — NVT MD**: Langevin thermostat, thermalization, LoggingHook to CSV.
