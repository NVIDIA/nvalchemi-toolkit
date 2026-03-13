Intermediate Examples
=====================

These examples assume familiarity with the basic tier and introduce
the storage layer, performance monitoring, and more complex pipeline
patterns.

**01 — Multi-Stage Pipeline**: FusedStage composition, LoggingHook CSV output,
step-budget migration, fused hooks for global status monitoring.

**02 — Trajectory I/O**: Writing trajectories to Zarr, reading back with
DataLoader, round-trip validation.

**03 — NPT MD**: Pressure-controlled dynamics with the MTK barostat,
LJ stress computation, cell fluctuation monitoring.

**04 — Inflight Batching**: SizeAwareSampler, Mode 2 FusedStage run (batch=None),
system_id tracking, ConvergedSnapshotHook collecting results.

**05 — Safety and Monitoring**: NaNDetectorHook, MaxForceClampHook,
EnergyDriftMonitorHook, ProfilerHook — defensive MD patterns.
